import os
import cv2
import numpy as np
import joblib
import time
import psutil 

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "face_classifier.pkl")

model = joblib.load(MODEL_PATH)
cap = cv2.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0
psutil.cpu_percent()


while True:
    ret, frame = cap.read()
    if not ret:
        continue

   
    new_frame_time = time.time()


    face = cv2.resize(frame,(64,64))
    x = face.flatten().reshape(1,-1)
    pred = model.predict(x)[0]


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps_text = f"FPS: {int(fps)}"


    cpu_usage = psutil.cpu_percent()
    cpu_text = f"CPU: {cpu_usage}%"


    cv2.putText(frame, pred, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    

    cv2.putText(frame, fps_text, (30,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    

    cv2.putText(frame, cpu_text, (30,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Face Classifier", frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()
