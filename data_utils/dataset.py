import cv2
import mediapipe as mp
import csv
import os
import pandas as pd

# === Nastavenie ===
FILENAME = "data.csv"
HEADERS = []
for i in range(21):
    HEADERS += [f"x{i}", f"y{i}", f"z{i}"]
HEADERS += ["label", "hand", "sample_id"]

# Vytvor súbor ak neexistuje
if not os.path.exists(FILENAME):
    with open(FILENAME, "w", newline="") as f:
        csv.writer(f).writerow(HEADERS)

# Načítaj existujúce dáta (kvôli číslovaniu)
data = pd.read_csv(FILENAME) if os.path.exists(FILENAME) else pd.DataFrame()

# === Funkcie ===
def save_sample(landmarks, label, hand, sample_name):
    row = landmarks + [label, hand, sample_name]
    with open(FILENAME, "a", newline="") as f:
        csv.writer(f).writerow(row)

def get_next_sample_id(label):
    if len(data) > 0:
        return (data["label"] == label).sum() + 1
    return 1

# === MediaPipe ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

print("🖐️ Možné gesta: Fist, Open, Thumb_Up, Thumb_Down, Peace, Rock, OK, Pinch")
print("➡️ Napíš gesto (napr. 'Fist'), potom stláčaj SPACE pre uloženie.")
print("➡️ Stlač C pre zmenu gesta alebo ESC pre ukončenie.\n")

current_label = input("Zadaj gesto: ").strip()
sample_counter = get_next_sample_id(current_label)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                detected_hand = hand_info.classification[0].label
                hand_label = "Left" if detected_hand == "Right" else "Right"

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                key = cv2.waitKey(1) & 0xFF

                if key == 32:  # SPACE
                    sample_id = f"{current_label}_{sample_counter:03d}"
                    save_sample(landmarks, current_label, hand_label, sample_id)
                    print(f"💾 Uložené: {sample_id} ({hand_label})")
                    sample_counter += 1

                elif key == ord("c"):
                    current_label = input("\nZadaj nové gesto: ").strip()
                    sample_counter = get_next_sample_id(current_label)
                    print(f"👉 Aktuálne gesto: {current_label}")

                elif key == 27:
                    print("👋 Ukončenie zberu dát.")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

        cv2.putText(frame, f"Gesto: {current_label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Gesture Data Collection", frame)

cap.release()
cv2.destroyAllWindows()
