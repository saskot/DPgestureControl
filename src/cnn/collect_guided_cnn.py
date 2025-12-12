import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

# ========== SETTINGS ==========
OUTPUT_FILE = "data.csv"          # dataset file
SAMPLES_PER_GESTURE = 80          # number of samples per gesture

# LIST OF GESTURES (LABEL, HAND, DESCRIPTION)
GESTURES = [
    # LEFT HAND
    {"label": "FIST_L",     "hand": "Left",  "desc": "LEFT - Fist"},
    {"label": "OPEN_L",     "hand": "Left",  "desc": "LEFT - Open hand"},
    {"label": "PEACE_L",    "hand": "Left",  "desc": "LEFT - Peace V"},
    {"label": "OK_L",       "hand": "Left",  "desc": "LEFT - OK sign"},
    {"label": "POINT_L",    "hand": "Left",  "desc": "LEFT - Point finger"},
    {"label": "MIDDLE_L",   "hand": "Left",  "desc": "LEFT - Middle finger"},

    # RIGHT HAND
    {"label": "FIST_R",     "hand": "Right", "desc": "RIGHT - Fist"},
    {"label": "OPEN_R",     "hand": "Right", "desc": "RIGHT - Open hand"},
    {"label": "PEACE_R",    "hand": "Right", "desc": "RIGHT - Peace V"},
    {"label": "OK_R",       "hand": "Right", "desc": "RIGHT - OK sign"},
    {"label": "POINT_R",    "hand": "Right", "desc": "RIGHT - Point finger"},
    {"label": "MIDDLE_R",   "hand": "Right", "desc": "RIGHT - Middle finger"},

    # LIKE / DISLIKE LEFT HAND
    {"label": "LIKE_L",     "hand": "Left",  "desc": "LEFT - Like (thumb up)"},
    {"label": "DISLIKE_L",  "hand": "Left",  "desc": "LEFT - Dislike (thumb down)"},

    # LIKE / DISLIKE RIGHT HAND
    {"label": "LIKE_R",     "hand": "Right", "desc": "RIGHT - Like (thumb up)"},
    {"label": "DISLIKE_R",  "hand": "Right", "desc": "RIGHT - Dislike (thumb down)"},

    # UNKNOWN - ANY RANDOM / WRONG / NO HAND
    {"label": "UNKNOWN",    "hand": "Any",   "desc": "UNKNOWN - random gesture or no hand"},
]

# ========== INIT CAMERA + MEDIAPIPE ==========
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera could not be opened!")
    exit()

# ========== CSV HEADER / EXISTING COUNTS ==========
if not os.path.exists(OUTPUT_FILE):
    cols = []
    for i in range(21):
        cols += [f"x{i}", f"y{i}", f"z{i}"]
    cols.append("label")

    df = pd.DataFrame(columns=cols)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Created new dataset file {OUTPUT_FILE}.")
    existing_counts = {g["label"]: 0 for g in GESTURES}
else:
    print(f"Appending to existing file {OUTPUT_FILE}.")
    df = pd.read_csv(OUTPUT_FILE)
    if "label" in df.columns:
        existing_counts = df["label"].value_counts().to_dict()
    else:
        existing_counts = {}
    for g in GESTURES:
        existing_counts.setdefault(g["label"], 0)

print("Controls:")
print("  C = capture sample")
print("  P = go to next gesture (only after enough samples)")
print("  U = undo last sample for current gesture")
print("  ESC = exit")

current_gesture_index = 0
samples_collected = 0
ready_for_next = False  # True when SAMPLES_PER_GESTURE reached

def undo_last_sample_for_label(label):
    if not os.path.exists(OUTPUT_FILE):
        print("No file to undo from.")
        return False

    df_all = pd.read_csv(OUTPUT_FILE)
    if "label" not in df_all.columns or df_all.empty:
        print("File empty or no label column, cannot undo.")
        return False

    indices = df_all.index[df_all["label"] == label].tolist()
    if not indices:
        print(f"No samples with label {label} to undo.")
        return False

    last_idx = indices[-1]
    df_all = df_all.drop(index=last_idx)
    df_all.to_csv(OUTPUT_FILE, index=False)
    print(f"Undone last sample for {label}.")
    return True

def save_landmarks_row(landmarks, label):
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y, lm.z])

    if len(coords) != 63:
        print("Invalid landmark count, sample ignored.")
        return False

    row = {}
    for i in range(21):
        row[f"x{i}"] = coords[3 * i]
        row[f"y{i}"] = coords[3 * i + 1]
        row[f"z{i}"] = coords[3 * i + 2]
    row["label"] = label

    df_row = pd.DataFrame([row])
    df_row.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
    return True

def save_empty_unknown():
    row = {}
    for i in range(21):
        row[f"x{i}"] = 0.0
        row[f"y{i}"] = 0.0
        row[f"z{i}"] = 0.0
    row["label"] = "UNKNOWN"

    df_row = pd.DataFrame([row])
    df_row.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
    return True

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        if current_gesture_index >= len(GESTURES):
            print("All gestures collected. Done.")
            break

        gesture_info = GESTURES[current_gesture_index]
        label = gesture_info["label"]
        expected_hand = gesture_info["hand"]
        desc = gesture_info["desc"]
        total_for_label = existing_counts.get(label, 0)

        ret, frame = cap.read()
        if not ret:
            print("Error reading camera frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)

        text1 = f"Gesture {current_gesture_index+1}/{len(GESTURES)}: {label}"
        text2 = desc
        text3 = f"Session samples: {samples_collected}/{SAMPLES_PER_GESTURE}"
        text4 = f"Total in file: {total_for_label}"
        text5 = "C=capture | P=next | U=undo | ESC=exit"

        cv2.putText(frame, text1, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, text2, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, text3, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, text4, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
        cv2.putText(frame, text5, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if ready_for_next:
            cv2.putText(frame, "TARGET REACHED - PRESS P FOR NEXT GESTURE",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        detected_hand = None

        if result.multi_hand_landmarks and result.multi_handedness:
            hand_landmarks = result.multi_hand_landmarks[0]
            hand_class = result.multi_handedness[0].classification[0]
            detected_hand = hand_class.label

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"Detected: {detected_hand}",
                        (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Hand not detected",
                        (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Guided gesture collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("Stopped by user.")
            break

        # NEXT GESTURE
        if key in [ord('p'), ord('P')]:
            if ready_for_next:
                print(f"Gesture {label} finished. Moving to next.")
                current_gesture_index += 1
                samples_collected = 0
                ready_for_next = False
            else:
                print(f"Not enough samples for {label} yet ({samples_collected}/{SAMPLES_PER_GESTURE}).")
            continue

        # UNDO
        if key in [ord('u'), ord('U')]:
            success = undo_last_sample_for_label(label)
            if success:
                if samples_collected > 0:
                    samples_collected -= 1
                existing_counts[label] = max(existing_counts.get(label, 1) - 1, 0)
                ready_for_next = samples_collected >= SAMPLES_PER_GESTURE
            continue

        # CAPTURE
        if key in [ord('c'), ord('C')]:
            # SPECIAL CASE: UNKNOWN
            if label == "UNKNOWN":
                if result.multi_hand_landmarks:
                    # save whatever hand is there, any pose
                    hand_landmarks = result.multi_hand_landmarks[0]
                    ok = save_landmarks_row(hand_landmarks.landmark, "UNKNOWN")
                    if ok:
                        samples_collected += 1
                        existing_counts["UNKNOWN"] = existing_counts.get("UNKNOWN", 0) + 1
                        print(f"Saved UNKNOWN with hand, sample {samples_collected}/{SAMPLES_PER_GESTURE}")
                else:
                    # no hand -> save zeros
                    ok = save_empty_unknown()
                    if ok:
                        samples_collected += 1
                        existing_counts["UNKNOWN"] = existing_counts.get("UNKNOWN", 0) + 1
                        print(f"Saved UNKNOWN empty sample {samples_collected}/{SAMPLES_PER_GESTURE}")

                if samples_collected >= SAMPLES_PER_GESTURE:
                    ready_for_next = True
                    print(f"Target {SAMPLES_PER_GESTURE} samples reached for UNKNOWN. Press P to go next.")
                continue

            # normal gestures (not UNKNOWN)
            if not result.multi_hand_landmarks or not result.multi_handedness:
                print("No hand detected, sample ignored.")
                continue

            hand_landmarks = result.multi_hand_landmarks[0]
            hand_class = result.multi_handedness[0].classification[0]
            detected_hand = hand_class.label

            if detected_hand != expected_hand:
                print(f"Wrong hand: detected {detected_hand}, expected {expected_hand}. Sample ignored.")
                continue

            # bounding box check - allow small, block only extremely small
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            bbox_w = max(xs) - min(xs)
            bbox_h = max(ys) - min(ys)

            if bbox_w < 0.05 or bbox_h < 0.05:
                print(f"Hand too small (bbox {bbox_w:.2f} x {bbox_h:.2f}). Move slightly closer. Sample ignored.")
                continue

            ok = save_landmarks_row(hand_landmarks.landmark, label)
            if ok:
                samples_collected += 1
                existing_counts[label] = existing_counts.get(label, 0) + 1
                print(f"Saved sample {samples_collected}/{SAMPLES_PER_GESTURE} for {label} (total: {existing_counts[label]})")

                if samples_collected >= SAMPLES_PER_GESTURE:
                    ready_for_next = True
                    print(f"Target {SAMPLES_PER_GESTURE} samples reached for {label}. Press P to go next gesture.")

cap.release()
cv2.destroyAllWindows()
print("Program finished.")
