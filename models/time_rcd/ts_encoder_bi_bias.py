import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from jaxtyping import Float, Int
from einops import rearrange


class RMSNorm(nn.Module):
    """Root Mean Square Normalization layer."""

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.to(torch.float32).pow(2).mean(dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.scale * x_normed).type_as(x)


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding for injecting positional information."""

    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return freqs  # Shape: (seq_len, dim // 2)


class BinaryAttentionBias(nn.Module):
    """Binary Variate Attention for time series data."""

    def __init__(self,
                 num_heads: Int):
        super().__init__()
        self.num_heads = num_heads
        self.emd = nn.Embedding(2, num_heads)

    def forward(self,
                query_id: Int[torch.Tensor, "batch_size q_len"],
                kv_id: Int[torch.Tensor, "batch_size kv_len"],
                ) -> Float[torch.Tensor, "batch_size num_heads q_len kv_len"]:
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))
        ind = ind.unsqueeze(1)  # (batch_size, 1, q_len, kv_len)
        weight = rearrange(self.emd.weight, "two num_heads -> two num_heads 1 1")  # (2, num_heads, 1, 1)
        bias = ~ind * weight[:1] + ind * weight[1:]  # (batch_size, num_heads, q_len, kv_len)
        return bias


class MultiheadAttentionWithRoPE(nn.Module):
    """Multi-head Attention with Rotary Positional Encoding (RoPE), non-causal by default."""
    "========== NOtice that this applies BinaryAttentionBias ==========="

    def __init__(self, embed_dim, num_heads, num_features):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_features = num_features
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V, and output
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Binary attention bias for time series
        if num_features > 1:
            self.binary_attention_bias = BinaryAttentionBias(num_heads)

    def apply_rope(self, x, freqs):
        """Apply Rotary Positional Encoding to the input tensor."""
        B, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch"
        assert freqs.shape == (seq_len, embed_dim // 2), "freqs shape mismatch"

        # Reshape for rotation: split embed_dim into pairs
        x_ = x.view(B, seq_len, embed_dim // 2, 2)
        cos = freqs.cos().unsqueeze(0)  # (1, seq_len, embed_dim // 2, 1)
        sin = freqs.sin().unsqueeze(0)  # (1, seq_len, embed_dim // 2, 1)

        # Apply rotation to each pair
        x_rot = torch.stack(
            [
                x_[..., 0] * cos - x_[..., 1] * sin,
                x_[..., 0] * sin + x_[..., 1] * cos,
            ],
            dim=-1
        )
        return x_rot.view(B, seq_len, embed_dim)

    def forward(self, query, key, value, freqs, query_id=None, kv_id=None, attn_mask=None):
        """
        Forward pass for multi-head attention with RoPE.

        Args:
            query (Tensor): Shape (B, T, C)
            key (Tensor): Shape (B, T, C)
            value (Tensor): Shape (B, T, C)
            freqs (Tensor): RoPE frequencies, shape (T, embed_dim // 2)
            query_id (Tensor, optional): Shape (B, q_len), feature IDs for query
            kv_id (Tensor, optional): Shape (B, kv_len), feature IDs for key/value
            attn_mask (Tensor, optional): Shape (B, T), True for valid positions, False for padding.

        Returns:
            Tensor: Attention output, shape (B, T, C)
        """
        B, T, C = query.shape
        assert key.shape == (B, T, C) and value.shape == (B, T, C), "query, key, value shapes must match"

        # Project inputs to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Apply RoPE to Q and K
        Q_rot = self.apply_rope(Q, freqs)
        K_rot = self.apply_rope(K, freqs)

        # Reshape for multi-head attention
        Q_rot = Q_rot.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        K_rot = K_rot.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Prepare attention mask for padding
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        else:
            attn_mask = None

        if query_id is not None and kv_id is not None:
            # Add binary attention bias
            attn_bias = self.binary_attention_bias(query_id, kv_id)  # (B, num_heads, q_len, kv_len)
            scores = torch.matmul(Q_rot, K_rot.transpose(-2, -1)) / math.sqrt(
                self.head_dim)  # (B, num_heads, q_len, kv_len)
            scores += attn_bias
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads, q_len, kv_len)
            y = torch.matmul(attn_weights, V)  # (B, num_heads, q_len, hs)

        else:
            # Compute scaled dot-product attention (non-causal) without binary bias
            # for param in self.binary_attention_bias.parameters():
            #     param.requires_grad = False
            y = F.scaled_dot_product_attention(
                Q_rot, K_rot, V,
                attn_mask=attn_mask,
                is_causal=False  # Non-causal attention for encoder
            )  # (B, nh, T, hs)

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return y


class LlamaMLP(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048):
        super().__init__()
        self.hidden_size = d_model
        self.intermediate_size = dim_feedforward
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = F.gelu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class TransformerEncoderLayerWithRoPE(nn.Module):
    """Transformer Encoder Layer with RoPE and RMSNorm."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", num_features=1):
        super().__init__()
        self.self_attn = MultiheadAttentionWithRoPE(d_model, nhead, num_features)
        self.dropout = nn.Dropout(dropout)
        self.input_norm = RMSNorm(d_model)
        self.output_norm = RMSNorm(d_model)
        self.mlp = LlamaMLP(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, src, freqs, src_id=None, attn_mask=None):
        residual = src
        src = self.input_norm(src)
        src = self.self_attn(src, src, src, freqs, src_id, src_id, attn_mask=attn_mask)
        src = src + residual
        residual = src
        src = self.output_norm(src)
        src = self.mlp(src)
        src = residual + self.dropout2(src)
        return src


class CustomTransformerEncoder(nn.Module):
    """Stack of Transformer Encoder Layers."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, num_layers, num_features):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithRoPE(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                num_features=num_features
            ) for _ in range(num_layers)
        ])

    def forward(self, src, freqs, src_id=None, attn_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, freqs, src_id, attn_mask=attn_mask)
        return output


class TimeSeriesEncoder(nn.Module):
    """
    Time Series Encoder with PatchTST-like patching, RoPE.

    Args:
        d_model (int): Model dimension
        d_proj (int): Projection dimension
        patch_size (int): Size of each patch
        num_layers (int): Number of encoder layers
        num_heads (int): Number of attention heads
        d_ff_dropout (float): Dropout rate
        max_total_tokens (int): Maximum sequence length
        use_rope (bool): Use RoPE if True
        num_features (int): Number of features in the time series
        activation (str): "relu" or "gelu"

    Inputs:
        time_series (Tensor): Shape (batch_size, seq_len, num_features)
        mask (Tensor): Shape (batch_size, seq_len)

    Outputs:
        local_embeddings (Tensor): Shape (batch_size, seq_len, num_features, d_proj)
    """

    def __init__(self, d_model=2048, d_proj=512, patch_size=32, num_layers=6, num_heads=8,
                 d_ff_dropout=0.1, max_total_tokens=8192, use_rope=True, num_features=1,
                 activation="relu"):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_proj = d_proj
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff_dropout = d_ff_dropout
        self.max_total_tokens = max_total_tokens
        self.use_rope = use_rope
        self.num_features = num_features
        self.activation = activation

        # Patch embedding layer
        self.embedding_layer = nn.Linear(patch_size, d_model)

        if use_rope:
            # Initialize RoPE and custom encoder
            self.rope_embedder = RotaryEmbedding(d_model)
            self.transformer_encoder = CustomTransformerEncoder(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=d_ff_dropout,
                activation=activation,
                num_layers=num_layers,
                num_features=num_features
            )
        else:
            # Standard encoder without RoPE
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=d_ff_dropout,
                batch_first=True,
                activation=activation
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection layers
        self.projection_layer = nn.Linear(d_model, patch_size * d_proj)
        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'linear' in name:
                if self.activation == "relu":
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif self.activation == "gelu":
                    nn.init.kaiming_uniform_(param, nonlinearity='gelu')
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, time_series, mask):
        """Forward pass to generate local embeddings."""
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(-1)
        device = time_series.device
        B, seq_len, num_features = time_series.size()
        assert num_features == self.num_features, f"Number of features mismatch with data: {num_features} vs param: {self.num_features}"
        assert mask.size() == (B, seq_len), "Mask shape mismatch"

        # Pad sequence to be divisible by patch_size
        padded_length = math.ceil(seq_len / self.patch_size) * self.patch_size
        if padded_length > seq_len:
            pad_amount = padded_length - seq_len
            time_series = F.pad(time_series, (0, 0, 0, pad_amount), value=0)
            mask = F.pad(mask, (0, pad_amount), value=0)

        # Convert to patches
        num_patches = padded_length // self.patch_size
        total_length = num_patches * num_features
        patches = time_series.view(B, num_patches, self.patch_size, num_features)
        patches = patches.permute(0, 3, 1, 2).contiguous()  # (B, num_features, num_patches, patch_size)
        patches = patches.view(B, num_features * num_patches, self.patch_size)  # (B, L, patch_size)
        # Create feature IDs for patches
        feature_id = torch.arange(num_features, device=device).repeat_interleave(
            num_patches)  # (num_features * num_patches = L,)
        feature_id = feature_id.unsqueeze(0).expand(B, -1)  # (B, L)

        # Embed patches
        embedded_patches = self.embedding_layer(patches)  # (B, L, d_model)

        # Create patch-level mask
        mask = mask.view(B, num_patches, self.patch_size)
        patch_mask = mask.sum(dim=-1) > 0  # (B, num_patches)
        full_mask = patch_mask.unsqueeze(1).expand(-1, num_features, -1)  # (B, num_features, num_patches)
        full_mask = full_mask.reshape(B, num_features * num_patches)  # (B, L)

        # Generate RoPE frequencies if applicable
        if self.use_rope:
            freqs = self.rope_embedder(total_length).to(device)
        else:
            freqs = None

        # Encode sequence
        if num_features > 1:
            output = self.transformer_encoder(
                embedded_patches,
                freqs=freqs,
                src_id=feature_id,
                attn_mask=full_mask
            )
        else:
            output = self.transformer_encoder(
                embedded_patches,
                freqs=freqs,
                attn_mask=full_mask
            )

        # Extract and project local embeddings
        patch_embeddings = output  # (B, L, d_model)
        patch_proj = self.projection_layer(patch_embeddings)  # (B, L, patch_size * d_proj)
        local_embeddings = patch_proj.view(B, num_features, num_patches, self.patch_size, self.d_proj)
        local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4)  # (B, num_patches, patch_size, num_features, d_proj)
        local_embeddings = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :,
                           :]  # (B, seq_len, num_features, d_proj)

        return local_embeddings
