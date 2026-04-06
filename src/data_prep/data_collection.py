import cv2
import os
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

base_dir = project_root / "raw"
person_name = input("Name/Tag: ")
save_dir = base_dir / person_name

os.makedirs(save_dir, exist_ok=True)


cap = cv2.VideoCapture(0) 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

existing_files = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]

if existing_files:
    nums = [int(f.split("_")[1].split(".")[0]) for f in existing_files]
    count = max(nums) + 1
else:
    count = 0
max_images = 50 

def detect_faces_multi_angle(gray):
    faces_all = []

    faces_frontal = face_cascade.detectMultiScale(gray, 1.2, 6, minSize=(80,80))
    faces_all.extend(faces_frontal)

    faces_profile = profile_cascade.detectMultiScale(gray, 1.2, 6, minSize=(80,80))
    faces_all.extend(faces_profile)

    for angle in [-20, 20]:
        M = cv2.getRotationMatrix2D((gray.shape[1]//2, gray.shape[0]//2), angle, 1)
        rotated = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))
        faces_rot = face_cascade.detectMultiScale(rotated, 1.2, 6, minSize=(80,80))
        faces_all.extend(faces_rot)

    return faces_all

print(f"Collecting data for: {person_name}")
print("Press 'c' to capture image. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = detect_faces_multi_angle(gray) 

    display_frame = frame.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Face Capture", display_frame)
    
    key = cv2.waitKey(1)
    
    if key == ord('c'):
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            (x, y, w, h) = faces[0] 
        
            face_roi = frame[y:y+h, x:x+w]
            
            img_name = f"{save_dir}/img_{count:03d}.jpg"
            cv2.imwrite(img_name, face_roi)
            print(f"Image saved: {img_name}")
            count += 1
        else:
            print("No face detected to save!")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
