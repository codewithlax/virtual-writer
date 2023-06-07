import cv2
import time

import numpy as np

import HandTrackingModule as htm

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
img_canvas = np.zeros((480, 640, 3), np.uint8)

while True:
    # Time variables to setup fps:
    previous_time = 0
    current_time = 0
    # Choosing the webcam
    cap = cv2.VideoCapture(0)
    detector = htm.HandDetector(detection_con=0.7, track_con=0.7)
    x1, y1 = 0, 0
    while True:
        # The below method returns if the capture was successful (True or False) & the image. Since this is in a while loop,
        # the data is always read from the webcam.
        success, img = cap.read()
        img = detector.find_hands(cv2.flip(img, 1))
        landmark_list = detector.find_position(img)
        if len(landmark_list) != 0:
            # Check if tip of index finger and thumb are close enough:
            if (abs(landmark_list[4][1] - landmark_list[8][1]) < 20) and (
                    abs(landmark_list[4][2] - landmark_list[8][2]) < 20):
                x2, y2 = landmark_list[8][1], landmark_list[8][2]
                # print("open")
                if x1 == 0 and y1 == 0:
                    x1, y1 = x2, y2
                # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.line(img_canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
                x1, y1 = x2, y2
            elif landmark_list[4][2] > landmark_list[0][2]:
                img_canvas = np.zeros_like(img_canvas)
            else:
                print('closed')
                x1, y1 = 0, 0

        img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, img_canvas)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # To show the image in a window
        cv2.imshow("Image", img)
        # cv2.imshow("Canvas", img_canvas)
        cv2.waitKey(1)

    # # Read the webcam frame
    # ret, frame = cap.read()
    #
    # # Draw a circle on the frame
    # center = (frame.shape[1] // 2, frame.shape[0] // 2)  # Calculate the center of the frame
    # radius = 50
    # color = (0, 255, 0)  # Green color
    # thickness = 2
    # cv2.circle(frame, center, radius, color, thickness)
    #
    # # Display the frame
    # cv2.imshow('Webcam', frame)
    #
    # # Check for 'q' key press to exit
    # if cv2.waitKey(1) == ord('q'):
    #     break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
