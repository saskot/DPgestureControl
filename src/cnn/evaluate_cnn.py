import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# PATHS (project structure)
# code/
#   data/data.csv
#   models/gesture_cnn.h5
#   models/gesture_cnn_labels.pkl
#   reports/...
#   src/cnn/evaluate_cnn.py
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH   = os.path.join(BASE_DIR, "..", "..", "data", "data.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "..", "..", "models")
REPORTS_DIR = os.path.join(BASE_DIR, "..", "..", "reports")

MODEL_FILE  = os.path.join(MODELS_DIR, "gesture_cnn.h5")
LABELS_FILE = os.path.join(MODELS_DIR, "gesture_cnn_labels.pkl")

CONFUSION_MATRIX_FIG = os.path.join(REPORTS_DIR, "confusion_matrix.png")
CLASS_REPORT_TXT     = os.path.join(REPORTS_DIR, "classification_report.txt")

os.makedirs(REPORTS_DIR, exist_ok=True)

# =========================
# SAFETY PRINTS
# =========================
print("DATA_PATH:", DATA_PATH)
print("MODEL_FILE:", MODEL_FILE)
print("LABELS_FILE:", LABELS_FILE)
print("REPORTS_DIR:", REPORTS_DIR)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

if not os.path.exists(LABELS_FILE):
    raise FileNotFoundError(f"Labels not found: {LABELS_FILE}")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

feature_names = []
for i in range(21):
    feature_names += [f"x{i}", f"y{i}", f"z{i}"]

missing = [c for c in feature_names + ["label"] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

X_raw = df[feature_names].astype(float).values
y_str = df["label"].astype(str).values

# =========================
# RESHAPE + NORMALIZATION
# =========================
X = X_raw.reshape((-1, 21, 3))

def preprocess_batch(X_in):
    Xp = np.array(X_in, dtype=np.float32, copy=True)
    wrist = Xp[:, 0:1, :]
    Xp = Xp - wrist
    norms = np.linalg.norm(Xp, axis=2)  # (N,21)
    max_norm = np.max(norms, axis=1, keepdims=True)  # (N,1)
    max_norm[max_norm < 1e-6] = 1e-6
    Xp = Xp / max_norm[:, :, None]
    return Xp

X = preprocess_batch(X)

# =========================
# LABEL ENCODING (use saved order)
# =========================
label_payload = joblib.load(LABELS_FILE)
classes = [c.strip() for c in label_payload["classes"]]  # clean
class_to_idx = {c: i for i, c in enumerate(classes)}
num_classes = len(classes)

# clean dataset labels
y_str = np.array([s.strip() for s in y_str], dtype=str)

# filter out labels not present in trained classes (e.g., UNKNOWN mismatch)
mask = np.array([lbl in class_to_idx for lbl in y_str], dtype=bool)

missing_labels = sorted(set(y_str[~mask].tolist()))
if len(missing_labels) > 0:
    print("WARNING: These labels are in data.csv but NOT in model classes, they will be ignored:")
    print(missing_labels)

X = X[mask]
y_str = y_str[mask]

y_int = np.array([class_to_idx[lbl] for lbl in y_str], dtype=int)
y_onehot = tf.keras.utils.to_categorical(y_int, num_classes=num_classes)

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test, y_train_int, y_test_int = train_test_split(
    X,
    y_onehot,
    y_int,
    test_size=0.2,
    random_state=42,
    stratify=y_int
)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_FILE)

# =========================
# PREDICT
# =========================
y_pred_proba = model.predict(X_test, verbose=0)
y_pred_int = np.argmax(y_pred_proba, axis=1)

# =========================
# CLASSIFICATION REPORT
# =========================
report = classification_report(
    y_test_int,
    y_pred_int,
    target_names=classes,
    digits=3
)

print("=== CLASSIFICATION REPORT ===")
print(report)

with open(CLASS_REPORT_TXT, "w", encoding="utf-8") as f:
    f.write(report)

# =========================
# CONFUSION MATRIX (normalized)
# =========================
cm = confusion_matrix(y_test_int, y_pred_int)

def plot_confusion_matrix(cm, classes, normalize=True, title="Normalized confusion matrix"):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

plot_confusion_matrix(cm, classes, normalize=True)
plt.savefig(CONFUSION_MATRIX_FIG, dpi=150)
plt.close()

print(f"Saved confusion matrix -> {CONFUSION_MATRIX_FIG}")
print(f"Saved classification report -> {CLASS_REPORT_TXT}")
