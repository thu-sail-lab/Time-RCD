# Test the implementation
import time
from .evaluation.metrics import get_metrics_optimized, get_metrics
import numpy as np
# Generate test data
np.random.seed(42)
score = np.random.rand(10000)
labels = np.random.randint(0, 2, 10000)

# Compare performance
print("Testing original implementation...")
start = time.time()
metrics_original = get_metrics(score, labels)
time_original = time.time() - start
print(f"Original time: {time_original:.2f}s\n")

print("Testing optimized implementation...")
start = time.time()
metrics_optimized = get_metrics_optimized(score, labels)
time_optimized = time.time() - start
print(f"Optimized time: {time_optimized:.2f}s\n")

print(f"Speedup: {time_original/time_optimized:.2f}x")

# Verify results are similar
for key in metrics_original:
    orig_val = metrics_original[key]
    opt_val = metrics_optimized[key]
    if abs(orig_val - opt_val) > 1e-5:
        print(f"Warning: {key} differs: {orig_val} vs {opt_val}")
