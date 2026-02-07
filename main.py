import numpy as np
import pickle
import os

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
    # Falls du die Daten lokal lädst:
    # X_train, y_train = ... 
    # weights = train_lr_simple(X_train, y_train)
    # with open('fertige_gewichte.pkl', 'wb') as f:
    #     pickle.dump(weights, f)
    print("Training abgeschlossen.")