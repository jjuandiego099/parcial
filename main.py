import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon="🐾",
    layout="centered"
)

# ── Estilos ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0e0e0e;
    color: #f0ede6;
}

.stApp {
    background-color: #0e0e0e;
}

h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem !important;
    letter-spacing: 4px;
    color: #f0ede6;
    margin-bottom: 0 !important;
}

.subtitle {
    font-size: 0.95rem;
    color: #888;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

.upload-box {
    border: 1.5px dashed #333;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    background: #161616;
    margin-bottom: 1.5rem;
}

.result-box {
    background: linear-gradient(135deg, #1a1a1a, #222);
    border: 1px solid #2a2a2a;
    border-left: 4px solid #c8a96e;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
}

.result-label {
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #c8a96e;
    margin-bottom: 0.4rem;
}

.result-breed {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    letter-spacing: 2px;
    color: #f0ede6;
    line-height: 1;
}

.result-confidence {
    font-size: 0.9rem;
    color: #888;
    margin-top: 0.4rem;
}

.confidence-bar-bg {
    background: #2a2a2a;
    border-radius: 999px;
    height: 6px;
    margin-top: 0.8rem;
    overflow: hidden;
}

.confidence-bar-fill {
    background: linear-gradient(90deg, #c8a96e, #e8c98e);
    height: 6px;
    border-radius: 999px;
    transition: width 0.6s ease;
}

.top-preds {
    margin-top: 1.2rem;
    border-top: 1px solid #2a2a2a;
    padding-top: 1rem;
}

.top-pred-item {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: #aaa;
    padding: 0.25rem 0;
}

.stButton > button {
    background: #c8a96e;
    color: #0e0e0e;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    letter-spacing: 1px;
    width: 100%;
}

.stButton > button:hover {
    background: #e8c98e;
    color: #0e0e0e;
}

hr {
    border-color: #1e1e1e;
}
</style>
""", unsafe_allow_html=True)

# ── Título ───────────────────────────────────────────────────────────────────
st.markdown("<h1>DOG BREED</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='color:#c8a96e;margin-top:-1rem!important;'>CLASSIFIER</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Stanford Dogs · 120 Razas · CNN</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Cargar modelo ────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    if not os.path.exists("modelo_perros.keras"):
        st.error("⚠️ No se encontró 'modelo_perros.keras' en la carpeta actual.")
        return None
    return tf.keras.models.load_model("modelo_perros.keras")

@st.cache_resource
def cargar_clases():
    images_path = "stanford_dogs/Images"
    if os.path.exists(images_path):
        clases = [raza.split("-")[1] for raza in sorted(os.listdir(images_path))
                  if os.path.isdir(os.path.join(images_path, raza))]
        return clases
    return [f"Clase {i}" for i in range(120)]

model  = cargar_modelo()
clases = cargar_clases()

# ── Subir imagen ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Sube una foto de un perro",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_container_width=True)

    if st.button("🐾  IDENTIFICAR RAZA"):
        if model is None:
            st.error("Modelo no cargado.")
        else:
            with st.spinner("Analizando..."):
                img_resized = img.resize((224, 224))
                img_array  = np.array(img_resized).astype("float32") / 255.0
                img_array  = np.expand_dims(img_array, axis=0)

                predicciones = model.predict(img_array)[0]
                top5_idx     = predicciones.argsort()[-5:][::-1]

                raza_pred   = clases[top5_idx[0]].replace("_", " ")
                confianza   = predicciones[top5_idx[0]] * 100
                bar_width   = int(confianza)

            # Resultado principal
            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Raza identificada</div>
                <div class="result-breed">{raza_pred.upper()}</div>
                <div class="result-confidence">Confianza: {confianza:.1f}%</div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{bar_width}%"></div>
                </div>
                <div class="top-preds">
                    {''.join([
                        f'<div class="top-pred-item"><span>{clases[top5_idx[i]].replace("_"," ").title()}</span><span>{predicciones[top5_idx[i]]*100:.1f}%</span></div>'
                        for i in range(1, 5)
                    ])}
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-box">
        <p style="font-size:2.5rem;margin:0">🐕</p>
        <p style="color:#666;margin:0.5rem 0 0">Arrastra una imagen aquí o haz clic en el botón</p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center;color:#444;font-size:0.8rem;letter-spacing:2px'>STANFORD DOGS DATASET · CNN MODEL</p>", unsafe_allow_html=True)