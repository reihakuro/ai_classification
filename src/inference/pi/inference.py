import json
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

MODEL_PATH = "model_int8.tflite"
META_PATH  = "class_names.json"
IMG_PATH   = "test.jpg"

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)
class_names = meta["class_names"]
img_size = tuple(meta["img_size"])

interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()
in_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

img = cv2.imread(IMG_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (img_size[1], img_size[0]))
x = img.astype(np.float32) / 255.0

scale, zero = in_det["quantization"]
xq = (x / scale + zero).astype(np.uint8)
xq = np.expand_dims(xq, axis=0)

latencies = [] 

for i in range(200): 
    t0 = time.perf_counter()
    interpreter.set_tensor(in_det["index"], xq)
    interpreter.invoke()
    t1 = time.perf_counter()
    
    latency_ms = (t1 - t0) * 1000
    latencies.append(latency_ms)

yq = interpreter.get_tensor(out_det["index"])
scale_o, zero_o = out_det["quantization"]
y = (yq.astype(np.float32) - zero_o) * scale_o
pred = int(np.argmax(y[0]))

mean_latency = np.mean(latencies)
p50_latency = np.percentile(latencies, 50)
p90_latency = np.percentile(latencies, 90)
p99_latency = np.percentile(latencies, 99)

print("Predict:", class_names[pred])
print("\n--- Benchmark 200 lần ---")
print(f"Mean Latency : {mean_latency:.2f} ms")
print(f"P50 Latency  : {p50_latency:.2f} ms")
print(f"P90 Latency  : {p90_latency:.2f} ms")
print(f"P99 Latency  : {p99_latency:.2f} ms")
