import cv2
import os

# 1. Cấu hình thư mục và ID người dùng
# Thay đổi tên này cho từng người (VD: person_01, person_02, person_03)
base_dir = "../calibration"
person_name = input("Name/Tag: ")
save_dir = os.path.join(base_dir, person_name)

# Tạo thư mục nếu chưa tồn tại [cite: 30]
os.makedirs(save_dir, exist_ok=True)

# 2. Khởi tạo Camera và bộ phát hiện khuôn mặt
cap = cv2.VideoCapture(0) # 0 là camera mặc định
# Tải mô hình nhận diện khuôn mặt có sẵn của OpenCV [cite: 34]
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
max_images = 50 # Mục tiêu thu thập khoảng 30-50 ảnh 

def detect_faces_multi_angle(gray):
    faces_all = []

    # detect mặt thẳng
    faces_frontal = face_cascade.detectMultiScale(gray, 1.2, 6, minSize=(80,80))
    faces_all.extend(faces_frontal)

    # detect mặt nghiêng
    faces_profile = profile_cascade.detectMultiScale(gray, 1.2, 6, minSize=(80,80))
    faces_all.extend(faces_profile)

    # detect khi cúi/ngửa bằng cách xoay ảnh
    for angle in [-20, 20]:
        M = cv2.getRotationMatrix2D((gray.shape[1]//2, gray.shape[0]//2), angle, 1)
        rotated = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))
        faces_rot = face_cascade.detectMultiScale(rotated, 1.2, 6, minSize=(80,80))
        faces_all.extend(faces_rot)

    return faces_all

print(f"Đang bắt đầu thu thập dữ liệu cho: {person_name}")
print("Nhấn 'c' để chụp ảnh. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Chuyển sang ảnh xám để tăng tốc độ nhận diện [cite: 34]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # Phát hiện khuôn mặt
    faces = detect_faces_multi_angle(gray) 

    # Vẽ khung hình chữ nhật quanh mặt để dễ quan sát
    display_frame = frame.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Face Capture", display_frame)
    
    key = cv2.waitKey(1)
    
    # 3. Chức năng chụp ảnh [cite: 31]
    if key == ord('c'):
        # Nếu phát hiện được khuôn mặt thì mới lưu
        if len(faces) > 0:
            # Lấy khuôn mặt đầu tiên hoặc lớn nhất được tìm thấy
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            (x, y, w, h) = faces[0] 
            
            # Cắt vùng ảnh khuôn mặt (ROI - Region of Interest) [cite: 34]
            face_roi = frame[y:y+h, x:x+w]
            
            # Lưu ảnh vào thư mục
            img_name = f"{save_dir}/img_{count:03d}.jpg"
            cv2.imwrite(img_name, face_roi)
            print(f"Đã lưu: {img_name}")
            count += 1
        else:
            print("Không tìm thấy khuôn mặt nào để lưu!")

    # Nhấn 'q' để thoát hoặc dừng khi đủ số lượng
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
