import os
import cv2
import multiprocessing as mp
import numpy as np
import time
import psutil
import tflite_runtime.interpreter as tflite  

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "model_int8.tflite")

def camera_worker(frame_queue):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if not frame_queue.full():
            frame_queue.put(frame)

def classifier_worker(frame_queue, result_queue):
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=2)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            
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
            pred_text = f"Class: {pred_idx}" 

            result_queue.put((frame, pred_text))

def display_worker(result_queue):
    prev_frame_time = 0
    psutil.cpu_percent()
    
    while True:
        if not result_queue.empty():
            frame, pred = result_queue.get()
            
            new_frame_time = time.time()
            if prev_frame_time != 0:
                fps = 1 / (new_frame_time - prev_frame_time)
            else:
                fps = 0
            prev_frame_time = new_frame_time
            
            cpu_usage = psutil.cpu_percent()
            
            fps_text = f"FPS: {int(fps)}"
            cpu_text = f"CPU: {cpu_usage}%"

            cv2.putText(frame, str(pred), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, fps_text, (30,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(frame, cpu_text, (30,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.imshow("Multiprocess Classifier", frame)

        if cv2.waitKey(1) == 27:
            break

if __name__=="__main__":
    frame_queue = mp.Queue(maxsize=5)
    result_queue = mp.Queue(maxsize=5)

    p1 = mp.Process(target=camera_worker, args=(frame_queue,))
    p2 = mp.Process(target=classifier_worker, args=(frame_queue, result_queue))
    p3 = mp.Process(target=display_worker, args=(result_queue,))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()

