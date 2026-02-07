import streamlit as st
import numpy as np
import cv2
import pickle
import gzip
import os
from streamlit_drawable_canvas import st_canvas

# --- HILFSFUNKTIONEN ---
def softmax(t):
    exps = np.exp(t - np.max(t, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def train_lr_simple(X, y):
    X_b = np.hstack((np.ones((X.shape[0], 1)), X))
    theta = np.zeros((X_b.shape[1], 10))
    for i in range(50):
        probs = softmax(np.dot(X_b, theta))
        grad = np.dot(X_b.T, (probs - y)) / len(X_b)
        theta -= 0.5 * grad
    return theta

# --- MODEL LADEN (Caching für Speed) ---
@st.cache_resource
def get_model():
    # Wir suchen die Datei direkt im gleichen Verzeichnis wie die app.py
    aktueller_ordner = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(aktueller_ordner, 'fertige_gewichte.pkl')
    
    # Falls das fehlschlägt, versuchen wir es ohne Pfadangabe
    if not os.path.exists(pickle_file):
        pickle_file = 'fertige_gewichte.pkl'

    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    else:
        st.error("Gewichts-Datei nicht gefunden! Bitte stelle sicher, dass 'fertige_gewichte.pkl' hochgeladen wurde.")
        return None

weights = get_model()

# --- UI DESIGN ---
st.set_page_config(page_title="KI Wahrnehmung - Daniel Wirth", layout="centered")
st.title("🖊️ KI-Ziffernerkennung")
st.write("Seminarfach-Projekt: Wahrnehmung in der Robotik")

# Auswahl der Eingabemethode
methode = st.radio("Wähle eine Eingabemethode:", ("Live Zeichnen", "Bild hochladen"))

img_final = None

if methode == "Live Zeichnen":
    st.write("Zeichne eine Ziffer in das schwarze Feld:")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=30,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        
    )
    if canvas_result.image_data is not None:
        # Konvertierung: RGBA -> Graustufen -> 28x28
        img_raw = canvas_result.image_data.astype(np.uint8)
        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_RGBA2GRAY)
        img_final = cv2.resize(img_gray, (28, 28)).astype("float32") / 255

else:
    uploaded_file = st.file_uploader("Foto hochladen (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 0)
        img_res = cv2.resize(img, (28, 28))
        # Invertieren für Fotos (da meist schwarz auf weiß geschrieben wird)
        img_final = 1 - (img_res.astype("float32") / 255)

# --- AUSWERTUNG ---
if canvas_result.image_data is not None:
    # --- SCHRITT 1: VORVERARBEITUNG (Das "Gehirn" vor der KI) ---
    img = canvas_result.image_data.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
    # Bounding Box finden: Wo wurde wirklich gezeichnet?
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img_cropped = img[y:y+h, x:x+w]
        
        # Quadratisch machen & Padding (MNIST-Stil)
        size = max(w, h) + 30 
        final_img = np.zeros((size, size), dtype=np.uint8)
        offset_x, offset_y = (size - w) // 2, (size - h) // 2
        final_img[offset_y:offset_y+h, offset_x:offset_x+w] = img_cropped
        
        # Auf 28x28 skalieren und leicht weichzeichnen
        img_res = cv2.resize(final_img, (28, 28), interpolation=cv2.INTER_AREA)
        img_res = cv2.GaussianBlur(img_res, (3, 3), 0)
        
        # Normalisieren
        img_final = img_res.astype("float32") / 255.0
    else:
        img_final = None

    # --- SCHRITT 2: DEIN BESTEHENDER CODE (Anzeige & Prediction) ---
    if img_final is not None and weights is not None:
        # Prediction
        img_flat = img_final.flatten().reshape(1, -1)
        img_with_bias = np.hstack((np.ones((1, 1)), img_flat))
        prediction = np.argmax(np.dot(img_with_bias, weights))
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.write("### KI-Sicht")
            # Hier zeigen wir das 28x28 Bild, damit die Lehrer sehen, was die KI sieht
            st.image(img_final, width=150) 
        with col2:
            st.write("### Ergebnis")
            st.header(f"Zahl: {prediction}")
            
        with st.expander("Mathematik dahinter"):
            st.write("Die KI berechnet ein Punktprodukt aus der 784-Pixel-Matrix und den gelernten Gewichten.")
            st.write(f"Vektor-Dimension: {img_flat.shape}")









