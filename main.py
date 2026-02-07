import numpy as np
import pickle
import gzip
import os
import tensorflow.compat.v1 as tf

# TensorFlow 1.x Kompatibilität (wegen deines alten Codes)
tf.disable_v2_behavior()

print("--- SYSTEM START (Daniel Wirth - Seminarfach) ---")

# --- HILFSFUNKTIONEN ---
def softmax(t):
    exps = np.exp(t - np.max(t, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def train_lr_simple(X, y_raw):
    # Umwandlung der Labels in One-Hot-Encoding
    y = (np.arange(10) == y_raw[:, None]).astype(np.float32)
    X_b = np.hstack((np.ones((X.shape[0], 1)), X))
    theta = np.zeros((X_b.shape[1], 10))
    
    print(f"Training Logistic Regression mit {len(X)} Bildern...")
    for i in range(1000): # 100 Epochen für bessere Genauigkeit
        probs = softmax(np.dot(X_b, theta))
        grad = np.dot(X_b.T, (probs - y)) / len(X_b)
        theta -= 0.01 * grad
        if i % 20 == 0:
            print(f"  Schritt {i} von 1000...")
    return theta

# --- DATEN LADEN ---
pfad = r"c:\Users\Daniel\Downloads\USPS_Digit_Classification-master\USPS_Digit_Classification-master\mnist.pkl"

if not os.path.exists(pfad):
    # Falls der absolute Pfad auf einem anderen PC nicht stimmt, versuche relativen Pfad
    pfad = "mnist.pkl"

try:
    with gzip.open(pfad, 'rb') as f:
        save = pickle.load(f, encoding='latin1')
    train_data = save[0][0]
    train_labels = save[0][1]
    test_data = save[1][0]
    test_labels = save[1][1]
    print(f"Erfolgreich geladen: {pfad}")
except Exception as e:
    print(f"FEHLER beim Laden: {e}")
    exit()

# --- TRAINING ---
# Wir speichern das Ergebnis in der Variable 'weights_final'
weights_final = train_lr_simple(train_data[:50000], train_labels[:50000])

# Test der Genauigkeit
X_test_b = np.hstack((np.ones((test_data.shape[0], 1)), test_data))
test_preds = np.argmax(np.dot(X_test_b, weights_final), axis=1)
acc = np.mean(test_preds == test_labels)
print(f"Test Set Genauigkeit: {acc * 100:.2f}%")

# --- SPEICHERN FÜR DIE WEB-APP ---
# Dieser Teil behebt deinen NameError
dateiname_export = "fertige_gewichte.pkl"
try:
    print(f"Speichere Gewichte in '{dateiname_export}'...")
    with open(dateiname_export, 'wb') as f:
        pickle.dump(weights_final, f)
    print("--- EXPORT ERFOLGREICH ---")
    print("Du kannst diese Datei jetzt auf GitHub hochladen.")
except Exception as e:
    print(f"Fehler beim Speichern: {e}")

print("--- DEMO BEENDET ---")