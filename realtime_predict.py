# realtime_recognition.py
import cv2
import mediapipe as mp
import joblib
import pandas as pd
import os
import warnings
from colorama import Fore, Style

# Potlačenie varovaní z protobuf
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

MODEL_FILE = "gesture_model_with_features.pkl"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"{MODEL_FILE} neexistuje. Najprv spusti train_model.py a ulož model.")

# Načítanie modelu a feature názvov
payload = joblib.load(MODEL_FILE)
model = payload["model"]
feature_names = payload["features"]

# Kontrola správneho počtu featureov
if len(feature_names) != 21 * 3:
    raise ValueError("Očakával som 63 feature names (21 x,y,z). Niečo nesedí.")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kamera sa nepodarilo otvoriť!")
    exit()

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Chyba pri čítaní rámu z kamery")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        gesture = "No Hand"
        confidence = 0.0

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                try:
                    # Pripravíme DataFrame s rovnakými stĺpcami ako pri tréningu
                    landmarks_df = pd.DataFrame([landmarks], columns=feature_names)

                    # Predikcia + pravdepodobnosti
                    probs = model.predict_proba(landmarks_df)[0]
                    pred_index = probs.argmax()
                    gesture = model.classes_[pred_index]
                    confidence = probs[pred_index] * 100

                    CONF_THRESHOLD = 80  
                    if confidence < CONF_THRESHOLD:
                        gesture = "Unknown"

                    # Farebný výpis do konzoly podľa istoty
                    if confidence > 80:
                        color = Fore.GREEN
                    elif confidence > 50:
                        color = Fore.YELLOW
                    else:
                        color = Fore.RED

                    print(f"{color}➡ Gesto: {gesture} ({confidence:.2f} %){Style.RESET_ALL}")

                except Exception as e:
                    gesture = "Error"
                    print("Chyba pri predikcii:", e)

                # Výpis do obrazu
                cv2.putText(frame, f"{gesture} ({confidence:.1f}%)",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0) if confidence > 80 else (0, 255, 255) if confidence > 50 else (0, 0, 255),
                            2, cv2.LINE_AA)
        else:
            cv2.putText(frame, gesture, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Realtime Gesture Recognition", frame)

        # Stlačenie ESC ukončí program
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
