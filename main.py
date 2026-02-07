import numpy as np
import pickle
import os
import gzip

# Einfache Logistische Regression Training
def train_lr_simple(X, y_raw):
    # One-Hot Encoding für 10 Klassen (0-9)
    y = np.zeros((y_raw.size, 10))
    y[np.arange(y_raw.size), y_raw.astype(int)] = 1
    
    # Bias-Term hinzufügen (Spalte mit Einsen)
    X_b = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # Gewichte initialisieren (784 Pixel + 1 Bias = 785 Zeilen, 10 Spalten)
    theta = np.zeros((X_b.shape[1], 10))
    
    # Hyperparameter
    learning_rate = 0.1 # Niedriger für mehr Präzision
    iterations = 1000   # Ausreichend für gute Konvergenz
    
    print("Starte Training...")
    for i in range(iterations):
        # Lineare Kombination
        z = np.dot(X_b, theta)
        
        # Softmax für Wahrscheinlichkeiten
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        # Gradientenberechnung
        gradient = np.dot(X_b.T, (probs - y)) / y_raw.size
        theta -= learning_rate * gradient
        
        if i % 100 == 0:
            print(f"Iteration {i} von {iterations}...")
            
    return theta

# Beispielhafter Aufruf (Hier deine MNIST Daten einfügen)
if __name__ == "__main__":
    mnist_path = os.path.join(os.path.dirname(__file__), 'mnist.pkl')
    
    if os.path.exists(mnist_path):
        print("Versuche Daten zu laden...")
        try:
            # Wir öffnen die Datei mit gzip.open statt mit dem normalen open
            with gzip.open(mnist_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1') 
                # MNIST ist oft so strukturiert: ((train_img, train_lbl), (val_img, val_lbl), (test_img, test_lbl))
                train_set, valid_set, test_set = data
                X_train, y_train = train_set
        except Exception as e:
            # Falls es doch kein Gzip war, probieren wir es normal
            print(f"Gzip fehlgeschlagen, probiere Standard-Pickle... Fehler: {e}")
            with open(mnist_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                X_train, y_train = data[0]

        print(f"Daten erfolgreich geladen! {X_train.shape[0]} Bilder bereit.")

        # 2. Modell trainieren
        weights = train_lr_simple(X_train, y_train)

        # 3. Gewichte speichern
        with open('fertige_gewichte.pkl', 'wb') as f:
            pickle.dump(weights, f)
        
        print("✅ 'fertige_gewichte.pkl' wurde erstellt!")
    else:
        print(f"❌ Fehler: {mnist_path} wurde nicht gefunden!")