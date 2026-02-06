import streamlit as st
import numpy as np
import cv2
import pickle
import gzip
import os

# --- HILFSFUNKTIONEN (wie in main.py) ---
def softmax(t):
    exps = np.exp(t - np.max(t, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def train_lr_simple(X, y):
    X_b = np.hstack((np.ones((X.shape[0], 1)), X))
    theta = np.zeros((X_b.shape[1], 10))
    for i in range(50): # Kurzes Training für die Web-Demo
        probs = softmax(np.dot(X_b, theta))
        grad = np.dot(X_b.T, (probs - y)) / len(X_b)
        theta -= 0.5 * grad
    return theta

# --- SEITEN-DESIGN ---
st.set_page_config(page_title="KI Wahrnehmung - Daniel Wirth", layout="centered")
st.title("🖊️ KI-Ziffernerkennung")
st.write("Projekt von Daniel Wirth – Fokus: Wahrnehmung in der Robotik")

# --- MODEL LADEN / TRAINIEREN ---
@st.cache_resource # Verhindert, dass bei jedem Klick neu trainiert wird
def get_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(script_dir, 'mnist.pkl')
    with gzip.open(pickle_file, 'rb') as f:
        save = pickle.load(f, encoding='latin1')
    
    tr_d = save[0][0]
    tr_l_raw = save[0][1]
    # One-Hot Encoding
    tr_l = (np.arange(10) == tr_l_raw[:,None]).astype(np.float32)
    
    # Modell trainieren
    weights = train_lr_simple(tr_d[:10000], tr_l[:10000])
    return weights

weights = get_model()

# --- UPLOAD BEREICH ---
uploaded_file = st.file_uploader("Lade ein Foto deiner Ziffer hoch (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Bild einlesen
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 0)
    
    # Vorverarbeitung (Preprocessing)
    img_res = cv2.resize(img, (28, 28))
    img_final = 1 - np.array(img_res, "float32") / 255
    img_flat = img_final.flatten().reshape(1, -1)
    img_with_bias = np.hstack((np.ones((1, 1)), img_flat))
    
    # Vorhersage (Prediction)
    prediction = np.argmax(np.dot(img_with_bias, weights))
    
    # Anzeige im Browser
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original-Foto", use_container_width=True)
    with col2:
        st.image(img_final, caption="KI-Wahrnehmung (28x28)", use_container_width=True, clamp=True)
    
    st.header(f"Ergebnis: Die KI erkennt eine **{prediction}**")

    # Zusatz-Info für die Prüfer
    with st.expander("Technische Details anzeigen"):
        st.write("Hier siehst du die numerische Matrix, die die KI verarbeitet:")
        st.dataframe(img_final)