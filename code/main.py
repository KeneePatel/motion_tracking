import cv2

cap = cv2.VideoCapture(-1)

while True:
    timer = cv2.getTickCount()
    success, img = cap.read()
    
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    cv2.putText(img, str(int(fps)), (16, 40), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (0, 155, 255), 2)
    cv2.imshow("Capture", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
