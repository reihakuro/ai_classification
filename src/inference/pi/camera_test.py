import cv2

cap = cv2.VideoCapture(0)

cap.read() 

ret, frame = cap.read()

if ret:
    cv2.imwrite("test.jpg", frame)
    print("Works")
else:
    print("Something went wrong :(")


cap.release()
