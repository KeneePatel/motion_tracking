import cv2

def motion_detection():
    cap = cv2.VideoCapture(-1)

    first_frame = None
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

    def preprocess(frame):
        greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian_frame = cv2.GaussianBlur(greyscale_frame, (21,21),0)
        blur_frame = cv2.blur(gaussian_frame, (5,5))
        return blur_frame


    while True:
        text = ''
        timer = cv2.getTickCount()
        success, frame = cap.read()

        greyscale_image = preprocess(frame)

        if first_frame is None:
            first_frame = greyscale_image 

        frame_delta = cv2.absdiff(first_frame, greyscale_image)
        thresh_image = cv2.threshold(frame_delta, 40, 255, cv2.THRESH_BINARY)[1]
        dilate_image = cv2.dilate(thresh_image, None, iterations=5)

        cnt, _ = cv2.findContours(dilate_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnt:
            if cv2.contourArea(c) < 1500:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
            text = 'Motion Detected'
        
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        cv2.putText(frame, str(int(fps)), (16, 30), font, 0.7, (0, 155, 255), 2)
        cv2.putText(frame, text, (16, 60), font, 0.7, (0, 155, 255), 2)
        cv2.imshow('Capture Feed', frame)
        cv2.imshow('Threshold', dilate_image)
        cv2.imshow('Frame_delta', frame_delta)

        first_frame = greyscale_image 

        if cv2.waitKey(1) & 0xff == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    motion_detection()
