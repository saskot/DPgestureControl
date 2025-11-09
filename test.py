import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def finger_up(hand_landmarks, tip_id, dip_id, h, w):
    """Zistí, či je prst hore (tip vyššie ako kĺb)."""
    tip_y = int(hand_landmarks.landmark[tip_id].y * h)
    dip_y = int(hand_landmarks.landmark[dip_id].y * h)
    return tip_y < dip_y

def thumb_up_down(hand_landmarks, h, w):
    """Špeciálne pre palec (kontrola v osi X namiesto Y)."""
    tip_x = int(hand_landmarks.landmark[4].x * w)
    mcp_x = int(hand_landmarks.landmark[2].x * w)
    return tip_x < mcp_x  # True = palec hore, False = palec dole

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture = "No Hand"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                h, w, c = frame.shape
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # kontrola prstov (okrem palca)
                fingers = []
                fingers.append(finger_up(hand_landmarks, 8, 6, h, w))   # ukazovák
                fingers.append(finger_up(hand_landmarks, 12, 10, h, w)) # prostredník
                fingers.append(finger_up(hand_landmarks, 16, 14, h, w)) # prstenník
                fingers.append(finger_up(hand_landmarks, 20, 18, h, w)) # malíček

                # palec špeciálne
                thumb_up = thumb_up_down(hand_landmarks, h, w)

                # rozhodovanie gest
                if sum(fingers) == 0 and not thumb_up:
                    gesture = "Fist"
                elif sum(fingers) == 4 and thumb_up:
                    gesture = "Open Palm"
                elif fingers[0] and not any(fingers[1:]):
                    gesture = "Index Up"
                elif fingers[0] and fingers[1] and not any(fingers[2:]):
                    gesture = "Peace"
                elif thumb_up and sum(fingers) == 0:
                    gesture = "Thumb Up"
                elif not thumb_up and sum(fingers) == 0:
                    gesture = "Thumb Down"
                else:
                    gesture = "Other"

                # zobraz text
                cv2.putText(frame, gesture, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
