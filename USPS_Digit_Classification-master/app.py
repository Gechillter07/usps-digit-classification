import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pickle
import os
import cv2

# --- SEITEN-KONFIGURATION ---
st.set_page_config(page_title="Ziffernerkennung KI", layout="centered")

st.title("🔢 Handschriftliche Ziffernerkennung")
st.write("Zeichne eine Ziffer (0-9) mittig in das schwarze Feld.")

# --- MODELL LADEN ---
@st.cache_resource
def load_trained_model():
    # Sucht die Datei im gleichen Verzeichnis
    model_path = os.path.join(os.path.dirname(__file__), 'fertige_gewichte.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        return None

weights = load_trained_model()

if weights is None:
    st.error("❌ 'fertige_gewichte.pkl' nicht gefunden! Bitte lade die Datei auf GitHub hoch.")
    st.stop()

# --- CANVAS EINSTELLUNGEN ---
col_c, col_empty = st.columns([1, 1]) # Zentrierung des Canvas
with col_c:
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=18,  # Reduziert für bessere Skalierung
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True,
    )

# --- BILDVERARBEITUNG & PREDICTION ---
if canvas_result.image_data is not None:
    # 1. Konvertierung in Graustufen
    img = canvas_result.image_data.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    # 2. Bounding Box finden (Zentrierung)
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img_cropped = img[y:y+h, x:x+w]

        # 3. Großzügiges Padding (MNIST-Stil: Zahl macht ca. 70% des Bildes aus)
        # Wir fügen einen Rand hinzu, damit die Zahl nicht an den 28x28 Kanten klebt
        padding = int(max(w, h) * 0.4) 
        size = max(w, h) + padding * 2
        final_img = np.zeros((size, size), dtype=np.uint8)
        
        offset_x = (size - w) // 2
        offset_y = (size - h) // 2
        final_img[offset_y:offset_y+h, offset_x:offset_x+w] = img_cropped

        # 4. Skalierung auf 28x28 Pixel
        img_res = cv2.resize(final_img, (28, 28), interpolation=cv2.INTER_AREA)
        
        # 5. Leichte Glättung (ahmt den Anti-Aliasing Effekt von MNIST nach)
        img_res = cv2.GaussianBlur(img_res, (3, 3), 0)

        # 6. Normalisierung (0 bis 1)
        img_final = img_res.astype("float32") / 255.0
        
        # --- PREDICTION ---
        img_flat = img_final.flatten().reshape(1, -1)
        # Bias-Term hinzufügen (die 1 am Anfang des Vektors)
        img_with_bias = np.hstack((np.ones((1, 1)), img_flat))
        
        # Berechnung: Ergebnis = argmax(X * W)
        probabilities = np.dot(img_with_bias, weights)
        prediction = np.argmax(probabilities)

        # --- ANZEIGE ---
        st.divider()
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.write("### KI-Wahrnehmung")
            st.image(img_final, width=150, caption="Skaliertes 28x28 Bild")
            
        with res_col2:
            st.write("### Ergebnis")
            st.markdown(f"<h1 style='color: #FF4B4B; font-size: 80px;'>{prediction}</h1>", unsafe_allow_html=True)

        # Info für das Kolloquium
        with st.expander("Technisches Protokoll"):
            st.write(f"Original-Ausschnitt: {w}x{h} Pixel")
            st.write(f"Vektor-Input: 784 Merkmale + 1 Bias")
            st.write("Modell: Logistische Regression (Linearer Klassifikator)")

    else:
        st.info("Zeichne etwas, um die Erkennung zu starten.")

# Button zum Löschen (Streamlit Canvas löscht bei Klick auf Mülltonne automatisch)
if st.button("Verlauf löschen"):
    st.rerun()
