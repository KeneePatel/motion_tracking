import cv2
import numpy as np

def motion_detection():
    cap = cv2.VideoCapture(-1)

    # Font declaration
    font_num = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    font_text = cv2.FONT_HERSHEY_PLAIN

    # options for important variables
    threshold_value = 30
    threshold_area = 100*150
    dilate_iters = 20

    # number of buffer frames to keep for calculating the median
    buffer_frames_number = 5
    buffer_frames = []

    # setting the Frame Widht and Height of the camera to capture upon
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # function for preprocessing the input frame
    def preprocess(frame):
        greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian_frame = cv2.GaussianBlur(greyscale_frame, (21,21),0)
        blur_frame = cv2.blur(gaussian_frame, (5,5))
        return blur_frame

    # initial filling of buffer to have the buffer frames ready when going into the loop
    for _ in range(buffer_frames_number-1):
        success, frame = cap.read()
        blur_frame = preprocess(frame)
        buffer_frames.append(blur_frame)

    while True:
        # Declarations of variables along with resetting of variables
        count = 0
        text = ''
        timer = cv2.getTickCount()
        success, frame = cap.read()

        # preprocessing the recent input frame and adding it to buffer
        blur_frame = preprocess(frame)
        buffer_frames.append(blur_frame)

        # calculating the median of frames and then popping of the last frame in queue manner (FIFO)
        median = np.median(buffer_frames, axis=0).astype(dtype=np.uint8)
        buffer_frames.pop(0)

        # post processing for getting the contours
        frame_delta = cv2.absdiff(median, blur_frame)
        thresh_image = cv2.threshold(frame_delta, threshold_value, 255, cv2.THRESH_BINARY)[1]
        dilate_image = cv2.dilate(thresh_image, None, iterations=dilate_iters)

        # getting the contours
        cnt, _ = cv2.findContours(dilate_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # plotting the major contours
        for c in cnt:
            if cv2.contourArea(c) < threshold_area:
                continue
            (x, y, w, h) = cv2.boundingRect(c)

            # Mending the rectangular area if we decide to keep the dilate iteration high as it
                # can mess up the visualization
            if dilate_iters >= 10:
                if x > dilate_iters:
                    x += dilate_iters
                    w -= dilate_iters
                if y > dilate_iters:
                    y += dilate_iters
                    h -= dilate_iters
                w = w-dilate_iters if x+w < frame_width else w
                h = h-dilate_iters if y+h < frame_height else h

            # putting the text into the frame
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'object' + str(count+1), (x+10, y-10), font_text, 0.9, (0, 155, 255), 2)
            count+=1
            text = 'Motion Detected'
        
        # to calculate the fps
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # showing all the different frames in each of the windows
        cv2.putText(frame, str(int(fps)), (16, 30), font_num, 0.7, (0, 155, 255), 2)
        cv2.putText(frame, text, (16, 60), font_text, 1, (0, 155, 255), 2)
        cv2.imshow('Capture Feed', frame)
        cv2.imshow('Threshold', dilate_image)
        cv2.imshow('Frame_delta', frame_delta)
        cv2.imshow('Median', median)

        # Press q to get out of the loop
        if cv2.waitKey(1) & 0xff == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    motion_detection()