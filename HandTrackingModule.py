import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        self.mode = mode,
        self.max_hands = max_hands,
        self.detection_con = detection_con,
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for each_hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, each_hand, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0, draw=True):

        landmark_list = []
        if self.results.multi_hand_landmarks:
            each_hand = self.results.multi_hand_landmarks[hand_number]
            for id, landmarks in enumerate(each_hand.landmark):
                h, w, c = img.shape
                # landmark w.r.t our image's position:
                cx, cy = int(landmarks.x * w), int(landmarks.y * h)
                landmark_list.append([id, cx, cy])
                # Making a landmark visibly distinct, so that it can be used for custom requirements
                if draw:
                    # circle on our image with the co-ordinates, radius, color and filled.
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

        return landmark_list


def main():
    # Time variables to setup fps:
    previous_time = 0
    current_time = 0
    # Choosing the webcam
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        # The below method returns if the capture was successful (True or False) & the image. Since this is in a while loop,
        # the data is always read from the webcam.
        success, img = cap.read()
        img = detector.find_hands(img)
        landmark_list = detector.find_position(img)
        if len(landmark_list) != 0:
            print(landmark_list[4])

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        # To show the image in a window
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
