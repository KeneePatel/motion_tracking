import cv2

cap = cv2.VideoCapture(-1)

while True:
    success, img = cap.read()
    
    cv2.imshow("Capture", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
