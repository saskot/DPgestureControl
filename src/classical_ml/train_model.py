# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import os

FILENAME = "data.csv"
MODEL_FILE = "gesture_model_with_features.pkl"

# === 1) načítanie CSV ===
if not os.path.exists(FILENAME):
    raise FileNotFoundError(f"{FILENAME} neexistuje. Najprv nazbieraj dáta.")

data = pd.read_csv(FILENAME)

# odstránime prípadné index stĺpce, ktoré pandas pridá (napr. 'Unnamed: 0')
unnamed = [c for c in data.columns if c.startswith("Unnamed")]
if unnamed:
    data = data.drop(columns=unnamed)

# === 2) zostavíme očakávané názvy stĺpcov v správnom poradí ===
feature_names = []
for i in range(21):
    feature_names += [f"x{i}", f"y{i}", f"z{i}"]

# skontrolujeme, že súbor obsahuje label, hand, sample_id na konci
expected_tail = ["label", "hand", "sample_id"]
# ak názvy v csv iné (napr. "sample"), prispôsobíme
tail_present = data.columns[-3:].tolist()
if tail_present != expected_tail:
    # pokúsime sa mapovať bežné varianty
    # ak sú posledné 3 stĺpce správne, premenovať na očakávané
    data.columns = list(data.columns[:-3]) + expected_tail
    tail_present = data.columns[-3:].tolist()

# === 3) validácia prítomnosti feature stĺpcov ===
missing = [c for c in feature_names if c not in data.columns]
if missing:
    raise ValueError(f"Chýbajú tieto feature stĺpce v {FILENAME}: {missing}")

# === 4) priprava X, y ===
X = data[feature_names].astype(float)
y = data["label"].astype(str)

# === 5) split + tréning ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"✅ Presnosť: {acc:.3f}")

# === 6) uloženie modelu + feature names dohromady ===
payload = {"model": model, "features": feature_names}
joblib.dump(payload, MODEL_FILE)
print(f"💾 Model + feature names uložené do {MODEL_FILE}")
