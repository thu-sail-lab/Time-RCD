import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel, GPT2Config, GPT2Model


class TextEncoder(nn.Module):
    """
    文本编码器，基于LLaMA或GPT-2模型生成固定大小的嵌入表示。

    Args:
        model_name (str): 模型名称，支持 'llama' 或 'gpt2'，默认 'llama'。
        d_proj (int): 输出嵌入维度，默认 512。
        num_layers (int): Transformer层数，默认 6。
        device (str, optional): 计算设备，默认自动选择 ('cuda' 或 'cpu')。
    """

    def __init__(self, model_name='llama', d_proj=512, num_layers=6, device=None):
        super().__init__()
        # 自动选择设备
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model_name = model_name.lower()

        # 模型配置与加载
        if self.model_name == 'llama':
            self.model_type = 'llama'
            config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            config.num_hidden_layers = num_layers
            self.model = LlamaModel.from_pretrained("huggyllama/llama-7b", config=config)
            d_model = config.hidden_size  # 4096 for LLaMA-7B
        elif self.model_name == 'gpt2':
            self.model_type = 'gpt2'
            config = GPT2Config.from_pretrained('openai-community/gpt2')
            config.n_layer = num_layers
            self.model = GPT2Model.from_pretrained('openai-community/gpt2', config=config)
            d_model = config.n_embd  # 768 for GPT-2
        else:
            raise ValueError("Unsupported model_name. Choose 'llama' or 'gpt2'.")

        # 投影层
        self.projection = nn.Linear(d_model, d_proj)

        # 将模型移动到指定设备
        self.model.to(self.device)
        self.projection.to(self.device)

        # 参数初始化
        self._init_parameters()

    def _init_parameters(self):
        """初始化模型参数"""
        # 投影层权重使用Xavier初始化
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 0.0)

    def forward(self, input_ids, attention_mask):
        """
        前向传播，生成文本嵌入。

        Args:
            input_ids (torch.Tensor): 输入token ID，形状 (batch_size, seq_len)。
            attention_mask (torch.Tensor): 注意力掩码，形状 (batch_size, seq_len)，1为真实数据，0为填充。

        Returns:
            torch.Tensor: 嵌入表示，形状 (batch_size, d_proj)。
        """
        # 输入验证
        assert input_ids.size(0) == attention_mask.size(0), "Batch size mismatch between input_ids and attention_mask"
        assert input_ids.size(1) == attention_mask.size(
            1), "Sequence length mismatch between input_ids and attention_mask"

        # 将输入移动到指定设备
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # 推理时禁用梯度计算
        with torch.no_grad():
            if self.model_type == 'llama':
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state

        # 计算平均隐藏状态，忽略填充部分
        masked_hidden = hidden_states * attention_mask.unsqueeze(-1)  # (batch_size, seq_len, d_model)
        sum_hidden = masked_hidden.sum(dim=1)  # (batch_size, d_model)
        valid_counts = attention_mask.sum(dim=1, keepdim=True)  # (batch_size, 1)
        mean_hidden = sum_hidden / valid_counts.clamp(min=1e-9)  # (batch_size, d_model)

        # 投影到d_proj维度
        embedding = self.projection(mean_hidden)  # (batch_size, d_proj)
        return embedding


# 使用示例
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    attention_mask[1, 8:] = 0  # 模拟填充值

    # 初始化模型
    encoder = TextEncoder(model_name='gpt2', d_proj=256, num_layers=4).to(device)
    embedding = encoder(input_ids, attention_mask)
    print(f"Embedding shape: {embedding.shape}")  # (2, 256)
