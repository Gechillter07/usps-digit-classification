import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pickle
import os
import cv2

st.set_page_config(page_title="KI Ziffernerkennung", layout="centered")

# Modell Laden Funktion
@st.cache_resource
def load_model():
    # Pfad zur Datei im aktuellen Verzeichnis
    path = os.path.join(os.path.dirname(__file__), 'fertige_gewichte.pkl')
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Fehler beim Lesen der Datei: {e}")
            return None
    return None

weights = load_model()

st.title("🔢 KI Handschrift-Erkennung")
st.write("Zeichne eine Zahl zwischen 0 und 9.")

if weights is None:
    st.error("⚠️ Datei 'fertige_gewichte.pkl' fehlt oder ist fehlerhaft (LFS Problem?).")
    st.stop()

# Canvas
canvas_result = st_canvas(
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Graustufen & Bounding Box
    img = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    coords = cv2.findNonZero(img)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img_cropped = img[y:y+h, x:x+w]
        
        # Padding für MNIST-Stil (Zahl zentriert mit Rand)
        pad = int(max(w, h) * 0.4)
        size = max(w, h) + 2 * pad
        processed = np.zeros((size, size), dtype=np.uint8)
        processed[pad:pad+h, pad:pad+w] = img_cropped
        
        # Resize auf 28x28 und Blur
        img_28 = cv2.resize(processed, (28, 28), interpolation=cv2.INTER_AREA)
        img_28 = cv2.GaussianBlur(img_28, (3, 3), 0)
        
        # Normalisierung & Prediction
        img_final = img_28.reshape(1, 784).astype("float32") / 255.0
        img_with_bias = np.hstack((np.ones((1, 1)), img_final))
        
        prediction = np.argmax(np.dot(img_with_bias, weights))
        
        # Anzeige
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.image(img_28, width=150, caption="KI-Sicht (28x28)")
        with c2:
            st.header(f"Ergebnis: {prediction}")
    else:
        st.info("Bitte zeichne eine Zahl.")