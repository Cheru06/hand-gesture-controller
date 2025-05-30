import sys
sys.path.insert(0, r"C:\Users\PNVRAO\handgestures-env\Lib\site-packages")
import cv2
import mediapipe as mp
import pyautogui
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

screen_w, screen_h = pyautogui.size()

class HandGestureController:
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
        self.last_pos = None
        self.swipe_threshold = 0.05
        self.cooldown = 0.5
        self.last_trigger = time.time()

    def detect(self, frame):
        gesture = None
        click = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            # Convert to screen coordinates
            screen_x = int(index_tip.x * screen_w)
            screen_y = int(index_tip.y * screen_h)
            pyautogui.moveTo(screen_x, screen_y, duration=0.05)

            # Detect click if index and thumb are close
            distance = ((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2) ** 0.5
            if distance < 0.03:
                click = True

            # Swipe gestures
            current_pos = (index_tip.x, index_tip.y)
            if self.last_pos:
                dx = current_pos[0] - self.last_pos[0]
                dy = current_pos[1] - self.last_pos[1]
                if abs(dx) > self.swipe_threshold and abs(dx) > abs(dy):
                    gesture = "left" if dx > 0 else "right"
                elif abs(dy) > self.swipe_threshold and abs(dy) > abs(dx):
                    gesture = "down" if dy > 0 else "up"
            self.last_pos = current_pos

            return gesture, click, hand_landmarks
        else:
            self.last_pos = None
        return None, False, None

def trigger_action(gesture, click, controller):
    now = time.time()
    if click and (now - controller.last_trigger > controller.cooldown):
        pyautogui.click()
        controller.last_trigger = now
        print("Click")

    if gesture and (now - controller.last_trigger > controller.cooldown):
        if gesture == "up":
            pyautogui.scroll(300)
        elif gesture == "down":
            pyautogui.scroll(-300)
        elif gesture == "left":
            pyautogui.press("left")
        elif gesture == "right":
            pyautogui.press("right")
        controller.last_trigger = now
        print(f"Gesture: {gesture}")

def main():
    cap = cv2.VideoCapture(1)
    controller = HandGestureController()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror the image
        gesture, click, landmarks = controller.detect(frame)

        if landmarks:
            mp_drawing.draw_landmarks(
                frame, landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

        if gesture or click:
            trigger_action(gesture, click, controller)

        if gesture:
            cv2.putText(frame, gesture.upper(), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if click:
            cv2.putText(frame, "CLICK", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Hand Control with Gestures", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
