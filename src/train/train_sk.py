import os
import cv2
import numpy as np
from sklearn.svm import SVC
from pathlib import Path
import joblib

current_file = Path(__file__).resolve().parent
ROOT = current_file.parent.parent 

DATASET_DIR = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "face_classifier.pkl"

X = []
y = []

for person in os.listdir(DATASET_DIR):

    person_dir = os.path.join(DATASET_DIR, person)

    for img_name in os.listdir(person_dir):

        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img,(64,64))
        img = img.flatten()
        
        X.append(img)
        y.append(person)

model = SVC(kernel="linear", probability=True)
model.fit(X,y)

joblib.dump(model, MODEL_PATH)

print("Training complete")
