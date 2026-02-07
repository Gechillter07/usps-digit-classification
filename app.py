import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pickle
import os
import cv2

# --- KONFIGURATION ---
st.set_page_config(page_title="KI Ziffernerkennung", layout="centered")

@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), 'fertige_gewichte.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

weights = load_model()

# --- UI ---
st.title("🔢 KI Handschrift-Erkennung")
st.write("Zeichne eine Zahl (0-9) in das Feld.")

if weights is None:
    st.error("❌ 'fertige_gewichte.pkl' fehlt auf GitHub!")
    st.stop()

# Zeichenfeld
canvas_result = st_canvas(
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
    update_streamlit=True,
)

# --- VERARBEITUNG ---
if canvas_result.image_data is not None:
    # 1. Bild vorbereiten
    img = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    coords = cv2.findNonZero(img)
    
    if coords is not None:
        # Bounding Box & Zentrierung
        x, y, w, h = cv2.boundingRect(coords)
        img_cropped = img[y:y+h, x:x+w]
        
        # Padding für MNIST-Stil
        pad = int(max(w, h) * 0.4)
        size = max(w, h) + 2 * pad
        processed = np.zeros((size, size), dtype=np.uint8)
        processed[pad:pad+h, pad:pad+w] = img_cropped
        
        # Auf 28x28 skalieren
        img_28 = cv2.resize(processed, (28, 28), interpolation=cv2.INTER_AREA)
        img_28 = cv2.GaussianBlur(img_28, (3, 3), 0)
        
        # 2. KI Prediction
        img_final = img_28.reshape(1, 784).astype("float32") / 255.0
        img_with_bias = np.hstack((np.ones((1, 1)), img_final))
        prediction = np.argmax(np.dot(img_with_bias, weights))
        
        # --- TECHNISCHER TEIL (ANZEIGE) ---
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### KI-Wahrnehmung")
            st.image(img_28, width=150, caption="Zentriertes 28x28 Bild")
            
        with col2:
            st.write("### Ergebnis")
            st.markdown(f"<h1 style='color: #FF4B4B; font-size: 80px;'>{prediction}</h1>", unsafe_allow_html=True)

        # Der mathematische Insight für die Lehrer
        with st.expander("🛠️ Technische Details & Mathematik"):
            st.write("**Preprocessing:** Bounding-Box Extraktion & Bikubische Interpolation.")
            st.write(f"**Eingabevektor:** 785 Dimensionen (784 Pixel + 1 Bias-Term).")
            st.write("**Klassifikator:** Multinomiale Logistische Regression.")
            st.latex(r"P(y=i|x) = \frac{e^{x^T w_i}}{\sum_{j=0}^{9} e^{x^T w_j}}")
    else:
        st.info("Zeichne eine Zahl, um die Analyse zu starten.")
