import cv2
import numpy as np

def motion_detection():
    cap = cv2.VideoCapture(-1)

    first_frame = None
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

    threshold_value = 30
    threshold_area = 3000
    dilate_iters = 10

    buffer_frames_number = 9
    buffer_frames = []

    def preprocess(frame):
        greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian_frame = cv2.GaussianBlur(greyscale_frame, (21,21),0)
        blur_frame = cv2.blur(gaussian_frame, (5,5))
        return blur_frame

    for _ in range(buffer_frames_number-1):
        success, frame = cap.read()
        blur_frame = preprocess(frame)
        buffer_frames.append(blur_frame)

    while True:
        text = ''
        timer = cv2.getTickCount()
        success, frame = cap.read()

        blur_frame = preprocess(frame)
        buffer_frames.append(blur_frame)

        median = np.median(buffer_frames, axis=0).astype(dtype=np.uint8)
        buffer_frames.pop(0)

        frame_delta = cv2.absdiff(median, blur_frame)
        thresh_image = cv2.threshold(frame_delta, threshold_value, 255, cv2.THRESH_BINARY)[1]
        dilate_image = cv2.dilate(thresh_image, None, iterations=dilate_iters)

        cnt, _ = cv2.findContours(dilate_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnt:
            if cv2.contourArea(c) < threshold_area:
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
        cv2.imshow('Median', median)

        if cv2.waitKey(1) & 0xff == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    motion_detection()