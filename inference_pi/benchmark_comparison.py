import os
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter

MODELS = ["model_int8.tflite", "pruned_int8.tflite"]

print(f"\n{'='*85}")
print(f"{'Model Name':<20} | {'Size (KB)':<12} | {'Mean (ms)':<15} | {'P90 (ms)':<12} | {'FPS':<10}")
print(f"{'-'*85}")

for m in MODELS:
    if not os.path.exists(m):
        print(f"{m:<20} | {'File Not Found':<12} | {'-':<15} | {'-':<12} | {'-':<10}")
        continue

    itp = Interpreter(model_path=m, num_threads=4)
    itp.allocate_tensors()
    inp = itp.get_input_details()[0]
    
    x = np.zeros(inp["shape"], dtype=inp["dtype"])

    for _ in range(10):
        itp.set_tensor(inp["index"], x)
        itp.invoke()

    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        itp.set_tensor(inp["index"], x)
        itp.invoke()
        t1 = time.perf_counter()
        times.append((t1-t0) * 1000)

    file_size_kb = os.path.getsize(m) / 1024
    mean_time = np.mean(times)
    p90_time = np.percentile(times, 90)
    fps = 1000 / mean_time

    print(f"{m:<20} | {file_size_kb:>12.2f} | {mean_time:>15.2f} | {p90_time:>12.2f} | {fps:>10.2f}")

print(f"{'='*85}\n")
