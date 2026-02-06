import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
import os
import gzip
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# --- HILFSFUNKTIONEN ---
def reformat(labels):
    num_labels = 10
    return (np.arange(num_labels) == labels[:,None]).astype(np.float32)

def resize_and_scale(img, size, scale):
    img = cv2.resize(img, size)
    return 1 - np.array(img, "float32")/scale

def accuracy(y, t):
    return float(np.sum(y == t)) / max(len(y), 1)

def one_hot_encoding(t):
    return np.argmax(t, axis=1)

def add_ones(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))

def softmax(t):
    exps = np.exp(t - np.max(t, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# --- DATEN LADEN ---
def get_training_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(script_dir, 'mnist.pkl')
    print(f"Lade MNIST-Daten aus: {pickle_file}")
    try:
        with gzip.open(pickle_file, 'rb') as f:
            save = pickle.load(f, encoding='latin1')
    except:
        with open(pickle_file, 'rb') as f:
            save = pickle.load(f, encoding='latin1')
    return save[0][0], reformat(save[0][1]), save[0][1], \
           save[1][0], reformat(save[1][1]), save[1][1], \
           save[2][0], reformat(save[2][1]), save[2][1]

def process_usps_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_data = os.path.join(script_dir, "USPSdata", "Numerals")
    if not os.path.exists(path_to_data):
        print("INFO: USPS-Ordner nicht gefunden. Vergleich wird nur mit MNIST durchgeführt.")
        return None, None
    sz = (28,28)
    usps_img, usps_lab = [], []
    for i in range(10):
        label_dir = os.path.join(path_to_data, str(i))
        if os.path.exists(label_dir):
            for name in os.listdir(label_dir):
                if '.png' in name:
                    img = cv2.imread(os.path.join(label_dir, name), 0)
                    if img is not None:
                        usps_img.append(resize_and_scale(img, sz, 255).flatten())
                        usps_lab.append(i)
    return np.array(usps_img), np.array(usps_lab)

# --- MODELL 1: LOGISTIC REGRESSION ---
def train_lr(X, y):
    print("Training Logistic Regression...")
    X_b = add_ones(X)
    theta = np.zeros((X_b.shape[1], 10))
    for i in range(500):
        probs = softmax(np.dot(X_b, theta))
        grad = np.dot(X_b.T, (probs - y)) / len(X_b)
        theta -= 0.5 * grad
    return theta

# --- MODELL 2: NEURONALES NETZ (SNN) ---
def train_snn(X, y_raw):
    print("Training Single Layer Neural Network (SNN)...")
    graph = tf.Graph()
    with graph.as_default():
        tf_train = tf.placeholder(tf.float32, shape=(None, 784))
        tf_labels = tf.placeholder(tf.float32, shape=(None, 10))
        w = tf.Variable(tf.truncated_normal([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        logits = tf.matmul(tf_train, w) + b
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=logits))
        opt = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        prediction = tf.nn.softmax(logits)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for step in range(10000):
            sess.run(opt, feed_dict={tf_train: X[:5000], tf_labels: reformat(y_raw[:5000])})
        return sess.run(w), sess.run(b)

# --- NEU: EIGENE ZAHL ERKENNEN ---
def erkenne_eigene_datei(dateiname, model_weights):
    if not os.path.exists(dateiname):
        print(f"\nFEHLER: Die Datei '{dateiname}' wurde nicht im Ordner gefunden!")
        return

    # Bild laden (Graustufen)
    img = cv2.imread(dateiname, 0)
    # Größe anpassen
    img_resized = cv2.resize(img, (28, 28))
    # Invertieren (Weiß auf Schwarz) und Normalisieren
    img_final = 1 - np.array(img_resized, "float32") / 255
    # Flach machen
    img_flat = img_final.flatten().reshape(1, -1)
    # Bias-Spalte hinzufügen für LR
    img_with_bias = np.hstack((np.ones((1, 1)), img_flat))
    
    # Vorhersage
    vorhersage = np.argmax(np.dot(img_with_bias, model_weights))
    
    print(f"\n--- ERGEBNIS EIGENE ZAHL ---")
    print(f"Datei: {dateiname}")
    print(f"KI-Wahrnehmung sagt: Es ist eine {vorhersage}!")
    
    # Bild anzeigen
    plt.imshow(img_final, cmap='gray')
    plt.title(f"Erkannt als: {vorhersage}")
    plt.show()

# --- HAUPTPROGRAMM ---
def main():
    print("--- SYSTEM START (Daniel Wirth - Seminarfach) ---")
    
    # 1. Daten laden
    tr_d, tr_l, tr_l_r, va_d, va_l, va_l_r, te_d, te_l, te_l_r = get_training_data()
    usps_d, usps_l = process_usps_data()

    # 2. Logistic Regression trainieren
    w_lr = train_lr(tr_d[:10000], tr_l[:10000])
    acc_lr = accuracy(te_l_r, one_hot_encoding(np.dot(add_ones(te_d), w_lr)))
    print(f"Test Set Genauigkeit (Logistic Regression): {acc_lr*100:.2f}%")

    # 3. SNN trainieren
    w_snn, b_snn = train_snn(tr_d, tr_l_r)
    pred_snn = np.dot(te_d, w_snn) + b_snn
    acc_snn = accuracy(te_l_r, one_hot_encoding(pred_snn))
    print(f"Test Set Genauigkeit (SNN): {acc_snn*100:.2f}%")

    # 4. Eigene Ziffer testen
    # Stelle sicher, dass "meine_zahl.png" im selben Ordner liegt!
    erkenne_eigene_datei("meine_zahl.png", w_lr)

    print("\n--- DEMO BEENDET ---")

if __name__ == "__main__":
    main()
