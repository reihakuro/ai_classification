import os
import cv2
import numpy as np
import time
import psutil 
import tflite_runtime.interpreter as tflite 

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "model_int8.tflite")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]

cap = cv2.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0
psutil.cpu_percent()

while True:
    ret, frame = cap.read()
   
    capture_time = time.time()
    
    if not ret:
        continue

    
    new_frame_time = time.time()

    
    resized_frame = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(resized_frame, axis=0)

    if input_details[0]['dtype'] == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    else:
        input_data = input_data.astype(input_details[0]['dtype'])

    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_idx = np.argmax(output_data)
    pred = f"Class: {pred_idx}" 

    
    process_end_time = time.time()
    latency_ms = (process_end_time - capture_time) * 1000

    
    if prev_frame_time != 0:
        fps = 1 / (new_frame_time - prev_frame_time)
    else:
        fps = 0
    prev_frame_time = new_frame_time
    
    
    cpu_usage = psutil.cpu_percent()
    
    
    fps_text = f"FPS: {int(fps)}"
    cpu_text = f"CPU: {cpu_usage}%"
    latency_text = f"Latency: {int(latency_ms)} ms"

    cv2.putText(frame, pred, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, fps_text, (30,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.putText(frame, cpu_text, (30,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    cv2.putText(frame, latency_text, (30,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)

    cv2.imshow("Single Process Classifier", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
