import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
from collections import deque
import os

# Absolútna cesta k tomuto súboru (src/cnn/...)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cesty relatívne k rootu projektu
DATA_PATH   = os.path.join(BASE_DIR, "..", "..", "data", "data.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "..", "..", "models")
REPORTS_DIR = os.path.join(BASE_DIR, "..", "..", "reports")

# Potom už len používaš tieto:
FILENAME        = DATA_PATH
MODEL_FILE      = os.path.join(MODELS_DIR, "gesture_cnn.h5")
LABELS_FILE     = os.path.join(MODELS_DIR, "gesture_cnn_labels.pkl")
TRAINING_CURVES_FIG   = os.path.join(REPORTS_DIR, "training_curves.png")
CONFUSION_MATRIX_FIG  = os.path.join(REPORTS_DIR, "confusion_matrix.png")
CLASS_REPORT_TXT      = os.path.join(REPORTS_DIR, "classification_report.txt")


CONF_THRESHOLD = 0.85  # 85 %
SMOOTH_WINDOW = 7      # koľko posledných predikcií vyhladzujeme

# ===== 1) LOAD MODEL + LABELS =====
model = tf.keras.models.load_model(MODEL_FILE)
labels_data = joblib.load(LABELS_FILE)
classes = labels_data["classes"]  # list stringov v rovnakom poradí ako výstup modelu

print("Načítané triedy:", classes)

# mapovanie na “user friendly” názvy
DISPLAY_MAP = {
    "FIST_L": "FIST",
    "FIST_R": "FIST",
    "OPEN_L": "OPEN",
    "OPEN_R": "OPEN",
    "PEACE_L": "PEACE",
    "PEACE_R": "PEACE",
    "THUMB_UP_L": "LIKE",
    "THUMB_UP_R": "LIKE",
    "THUMB_DOWN_L": "DISLIKE",
    "THUMB_DOWN_R": "DISLIKE",
    # ...
    "UNKNOWN": "Unknown"
}

def preprocess_single_landmarks(landmarks):
    """
    landmarks: list 21 bodov, každý má x,y,z (MediaPipe format).
    Výstup: numpy array shape (1,21,3) po posune + normalizácii.
    """
    coords = []
    for lm in landmarks:
        coords.append([lm.x, lm.y, lm.z])
    arr = np.array(coords, dtype=np.float32)  # (21,3)

    # posun k zápästiu
    wrist = arr[0]
    arr = arr - wrist

    norms = np.linalg.norm(arr, axis=1)
    max_norm = np.max(norms)
    if max_norm < 1e-6:
        max_norm = 1e-6
    arr = arr / max_norm

    return arr.reshape(1, 21, 3)

# ===== 2) MEDIAPIPE SETUP =====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

history = deque(maxlen=SMOOTH_WINDOW)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # zrkadlenie (aby to bolo intuitívne)
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        h, w, c = frame.shape

        current_label_str = "No hand"
        current_conf = 0.0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # vykresli skeleton
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # preprocess
            X_input = preprocess_single_landmarks(hand_landmarks.landmark)

            # predikcia
            proba = model.predict(X_input, verbose=0)[0]  # (num_classes,)
            max_idx = int(np.argmax(proba))
            max_conf = float(proba[max_idx])

            raw_label = classes[max_idx]  # string
            current_conf = max_conf * 100.0

            # threshold + UNKNOWN logika
            if max_conf < CONF_THRESHOLD:
                current_label_str = "Unknown"
            elif raw_label == "UNKNOWN":
                current_label_str = "Unknown"
            else:
                current_label_str = DISPLAY_MAP.get(raw_label, raw_label)

            # pridaj do histórie (nepridávaj “No hand”)
            history.append(current_label_str)
        else:
            # ak nie je ruka, históriu môžeš vyčistiť
            history.clear()

        # vyhladzovanie – najčastejšie gesto v histórii
        if len(history) > 0:
            # spočítaj výskyt
            uniq, counts = np.unique(list(history), return_counts=True)
            stable_label = uniq[np.argmax(counts)]
        else:
            stable_label = current_label_str

        # overlay text
        text = f"{stable_label} ({current_conf:.1f}%)"
        cv2.putText(frame, text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Gesture CNN Realtime", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
