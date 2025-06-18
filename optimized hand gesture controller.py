import sys
sys.path.insert(0, r"C:\Users\PNVRAO\handgestures-env\Lib\site-packages")
import cv2
import mediapipe as mp
import pyautogui
import time
from collections import deque

sys.path.insert(0, r"C:\Users\PNVRAO\handgestures-env\Lib\site-packages")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class IndexFingerSwipeTracker:
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
        self.positions = deque(maxlen=5)  # for smoothing
        self.swipe_threshold_x = 0.03  # reduced for sensitivity
        self.swipe_threshold_y = 0.03
        self.cooldown = 0.6  # seconds between triggers
        self.last_trigger = time.time()

    def detect_gesture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        gesture = None
        hand_landmarks = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_tip = hand_landmarks.landmark[8]
            self.positions.append((index_tip.x, index_tip.y))

            if len(self.positions) == self.positions.maxlen:
                avg_dx = self.positions[-1][0] - self.positions[0][0]
                avg_dy = self.positions[-1][1] - self.positions[0][1]

                if abs(avg_dx) > self.swipe_threshold_x and abs(avg_dx) > abs(avg_dy):
                    gesture = "left" if avg_dx > 0 else "right"

                elif abs(avg_dy) > self.swipe_threshold_y and abs(avg_dy) > abs(avg_dx):
                    gesture = "down" if avg_dy > 0 else "up"

        else:
            self.positions.clear()

        return gesture, hand_landmarks

def trigger_key(gesture, tracker):
    now = time.time()
    if gesture and (now - tracker.last_trigger > tracker.cooldown):
        pyautogui.press(gesture)
        tracker.last_trigger = now

def main():
    cap = cv2.VideoCapture(0)

    # OPTIONAL: Reduce resolution for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = IndexFingerSwipeTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gesture, landmarks = tracker.detect_gesture(frame)

        if landmarks:
            mp_drawing.draw_landmarks(
                frame,
                landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 165, 255), thickness=2, circle_radius=5),  # orange points
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)  # cyan lines
            )

        if gesture:
            cv2.putText(frame, gesture.upper(), (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            trigger_key(gesture, tracker)

        cv2.imshow("Gesture Control (Smoothed & Colored)", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
