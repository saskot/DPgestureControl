import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import joblib
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

np.random.seed(42)
tf.random.set_seed(42)

# ===== 1) NAČÍTANIE DÁT =====
if not os.path.exists(FILENAME):
    raise FileNotFoundError(f"Dataset {FILENAME} neexistuje.")

df = pd.read_csv(FILENAME)

# Odstráň prípadné "Unnamed" indexové stĺpce
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Feature stĺpce (x0..z20)
feature_names = []
for i in range(21):
    feature_names.append(f"x{i}")
    feature_names.append(f"y{i}")
    feature_names.append(f"z{i}")

for name in feature_names:
    if name not in df.columns:
        raise ValueError(f"V dataset neexistuje stĺpec {name}. Skontroluj zber dát.")

X_raw = df[feature_names].values  # (N, 63)
y_str = df["label"].astype(str).values  # (N,)

print("Dostupné triedy (vrátane UNKNOWN):", np.unique(y_str))

# ===== 2) RESHAPE + NORMALIZÁCIA =====
# z (N,63) -> (N,21,3)
X = X_raw.reshape((-1, 21, 3))

def preprocess_batch(X_in):
    """Posun k zápästiu (index 0) + normalizácia podľa max vzdialenosti."""
    X_proc = np.copy(X_in)
    # posun k zápästiu
    wrist = X_proc[:, 0:1, :]   # (N,1,3)
    X_proc = X_proc - wrist     # posun

    # normovanie podľa najväčšej vzdialenosti
    norms = np.linalg.norm(X_proc, axis=2)  # (N,21)
    max_norm = np.max(norms, axis=1, keepdims=True)  # (N,1)
    max_norm[max_norm < 1e-6] = 1e-6
    X_proc = X_proc / max_norm[:, :, None]  # (N,21,3)

    return X_proc

X_norm = preprocess_batch(X)

# ===== 3) LABELY =====
# kategórie 0..C-1
y_cat = pd.Series(y_str).astype("category")
classes = list(y_cat.cat.categories)
y_int = y_cat.cat.codes.values  # (N,)
num_classes = len(classes)

print("Počet tried:", num_classes)
for i, c in enumerate(classes):
    count = np.sum(y_int == i)
    print(f"  {i}: {c} (n={count})")

y_onehot = tf.keras.utils.to_categorical(y_int, num_classes=num_classes)

# ===== 4) TRAIN/TEST SPLIT =====
X_train, X_test, y_train, y_test, y_train_int, y_test_int = train_test_split(
    X_norm, y_onehot, y_int,
    test_size=0.2,
    random_state=42,
    stratify=y_int
)

# ===== 5) CLASS WEIGHTS =====
class_weights_vals = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=y_train_int
)
class_weight = {i: w for i, w in enumerate(class_weights_vals)}
print("Class weights:", class_weight)

# ===== 6) MODEL =====
layers = tf.keras.layers
models = tf.keras.models
regularizers = tf.keras.regularizers

inputs = layers.Input(shape=(21, 3), name="input_landmarks")

x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="valid")(inputs)
x = layers.BatchNormalization()(x)

x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="valid")(x)
x = layers.BatchNormalization()(x)

x = layers.GlobalAveragePooling1D()(x)

x = layers.Dense(
    128,
    activation="relu",
    kernel_regularizer=regularizers.l2(1e-4)
)(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===== 7) CALLBACKS =====
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# ===== 8) TRÉNING =====
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight
)

# ===== 9) EVALUÁCIA =====
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"=== Final CNN test accuracy: {test_acc:.3f}, loss: {test_loss:.3f} ===")

# Predikcie pre metriky
y_pred_proba = model.predict(X_test)
y_pred_int = np.argmax(y_pred_proba, axis=1)
y_true_int = y_test_int  # už integer

# Classification report
report_str = classification_report(
    y_true_int,
    y_pred_int,
    target_names=classes,
    digits=3
)
print("=== Classification report ===")
print(report_str)

with open(CLASS_REPORT_TXT, "w", encoding="utf-8") as f:
    f.write(report_str)
print(f"Classification report uložený do {CLASS_REPORT_TXT}")

# ===== 10) GRAFY: TRAINING CURVES =====
hist = history.history

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(hist["accuracy"], label="train acc")
plt.plot(hist["val_accuracy"], label="val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(hist["loss"], label="train loss")
plt.plot(hist["val_loss"], label="val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")

plt.tight_layout()
plt.savefig(TRAINING_CURVES_FIG, dpi=150)
plt.close()
print(f"Training curves uložené do {TRAINING_CURVES_FIG}")

# ===== 11) CONFUSION MATRIX =====
cm = confusion_matrix(y_true_int, y_pred_int)

def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix"):
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_display = cm

    plt.figure(figsize=(8, 8))
    plt.imshow(cm_display, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm_display.max() / 2.0

    for i, j in itertools.product(range(cm_display.shape[0]), range(cm_display.shape[1])):
        plt.text(
            j, i,
            format(cm_display[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm_display[i, j] > thresh else "black"
        )

    plt.ylabel("Skutočná trieda")
    plt.xlabel("Predikovaná trieda")
    plt.tight_layout()

# ulož oba: nenormalizovanú a normalizovanú, tu dám normalizovanú
plot_confusion_matrix(cm, classes, normalize=True, title="Normalized confusion matrix")
plt.savefig(CONFUSION_MATRIX_FIG, dpi=150)
plt.close()
print(f"Confusion matrix uložený do {CONFUSION_MATRIX_FIG}")

# ===== 12) SAVE MODEL + LABELS =====
model.save(MODEL_FILE)
joblib.dump({"classes": classes}, LABELS_FILE)
print(f"Model uložený do {MODEL_FILE}")
print(f"Labels uložené do {LABELS_FILE}")


