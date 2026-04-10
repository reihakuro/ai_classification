import json
import time
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import tflite_runtime.interpreter as tflite


BROKER = "localhost"
PORT = 1883
TOPIC = "edgeai/face_classifier"
PUBLISH_INTERVAL = 0.5

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL_PATH = "model.tflite"
LABEL_MAP_PATH = "label_map.json"
UNKNOWN_THRESHOLD = 70.0
MIN_FACE_SIZE = 80
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def load_label_map(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    except Exception as e:
        print(f"Lỗi đọc label map: {e}")
        return {}

def main():

    detector = cv2.CascadeClassifier(CASCADE_PATH)
    label_map = load_label_map(LABEL_MAP_PATH)

    print(f"Loading: {MODEL_PATH}")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]
    is_floating_model = input_details[0]['dtype'] == np.float32

    client = mqtt.Client()
    client.connect(BROKER, PORT, 60)
    client.loop_start()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("No camera found or cannot be opened.")

    prev_time = time.time()
    last_publish = 0.0
    print("Ready!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
        )

        predictions = []

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            face_resized = cv2.resize(face_roi, (input_width, input_height))

            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(face_rgb, axis=0)

            if is_floating_model:
                input_data = input_data.astype(np.float32) / 255.0

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            label_id = int(np.argmax(output_data[0]))
            confidence = float(output_data[0][label_id]) * 100

            if confidence >= UNKNOWN_THRESHOLD:
                label = label_map.get(label_id, "unknown")
            else:
                label = "unknown"

            predictions.append({
                "label": label,
                "confidence": round(confidence, 2),
                "bbox": [int(x), int(y), int(w), int(h)]
            })

            text = f"{label} ({confidence:.1f}%)"
            color = (0, 255, 0) if label != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        if current_time - last_publish >= PUBLISH_INTERVAL:
            payload = {
                "timestamp": current_time,
                "has_face": len(predictions) > 0,
                "face_count": len(predictions),
                "predictions": predictions,
                "fps": round(fps, 2)
            }
            client.publish(TOPIC, json.dumps(payload, ensure_ascii=False))
            last_publish = current_time

        cv2.imshow("TFLite Face Classifier MQTT", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    client.loop_stop()
    client.disconnect()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()