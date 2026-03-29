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
    ruta_modelo = os.path.join(os.getcwd(), "model.keras")
    if not os.path.exists(ruta_modelo):
        st.error(f"❌ No se encontró el archivo: {ruta_modelo}")
        return None

    import builtins
    import tensorflow as tf
    builtins.tf = tf  # ← inyecta tf globalmente para que Lambda lo encuentre

    return tf.keras.models.load_model(
        ruta_modelo,
        custom_objects={"tf": tf},
        safe_mode=False
    )
modelo=cargar_modelo()
    
clases = [
    'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih', 'Blenheim_spaniel',
    'papillon', 'toy_terrier', 'Rhodesian_ridgeback', 'Afghan_hound', 'basset', 'beagle',
    'bloodhound', 'bluetick', 'black', 'Walker_hound', 'English_foxhound', 'redbone',
    'borzoi', 'Irish_wolfhound', 'Italian_greyhound', 'whippet', 'Ibizan_hound',
    'Norwegian_elkhound', 'otterhound', 'Saluki', 'Scottish_deerhound', 'Weimaraner',
    'Staffordshire_bullterrier', 'American_Staffordshire_terrier', 'Bedlington_terrier',
    'Border_terrier', 'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier',
    'Norwich_terrier', 'Yorkshire_terrier', 'wire', 'Lakeland_terrier', 'Sealyham_terrier',
    'Airedale', 'cairn', 'Australian_terrier', 'Dandie_Dinmont', 'Boston_bull',
    'miniature_schnauzer', 'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier',
    'Tibetan_terrier', 'silky_terrier', 'soft', 'West_Highland_white_terrier', 'Lhasa',
    'flat', 'curly', 'golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever',
    'German_short', 'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter',
    'Brittany_spaniel', 'clumber', 'English_springer', 'Welsh_springer_spaniel',
    'cocker_spaniel', 'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke',
    'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old_English_sheepdog',
    'Shetland_sheepdog', 'collie', 'Border_collie', 'Bouvier_des_Flandres', 'Rottweiler',
    'German_shepherd', 'Doberman', 'miniature_pinscher', 'Greater_Swiss_Mountain_dog',
    'Bernese_mountain_dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull_mastiff',
    'Tibetan_mastiff', 'French_bulldog', 'Great_Dane', 'Saint_Bernard', 'Eskimo_dog',
    'malamute', 'Siberian_husky', 'affenpinscher', 'basenji', 'pug', 'Leonberg',
    'Newfoundland', 'Great_Pyrenees', 'Samoyed', 'Pomeranian', 'chow', 'keeshond',
    'Brabancon_griffon', 'Pembroke', 'Cardigan', 'toy_poodle', 'miniature_poodle',
    'standard_poodle', 'Mexican_hairless', 'dingo', 'dhole', 'African_hunting_dog'
]



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
        if modelo is None:
            st.error("Modelo no cargado.")
        else:
            with st.spinner("Analizando..."):
                img_resized = img.resize((150, 150))
                img_array  = np.array(img_resized).astype("float32") / 255.0
                img_array  = np.expand_dims(img_array, axis=0)

                predicciones = modelo.predict(img_array)[0]
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
