import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Clasificador de Flores IA", layout="centered")

# --- ENCABEZADO OBLIGATORIO (INFORMACIÓN DEL ESTUDIANTE) ---
st.title("Clasificación de Flores con Inteligencia Artificial")
st.markdown("### **Asignatura:** IS-701 - Inteligencia Artificial (Campus Comayagua)")
st.markdown("### **Estudiante:** Edwin Eduardo Guzmán Ramos")
st.markdown("### **Número de Cuenta:** 20211930058")
st.write("Suba una imagen para clasificarla mediante el modelo MobileNetV2 adaptado.")
st.write("---") # Línea divisoria visual

IMG_SIZE = (224, 224)

# --- CONFIGURACIÓN DE RUTAS DEL MODELO EN GITHUB ---
# Apuntamos a la carpeta exacta "models" que descargaste de Colab
MODEL_DIR = Path("models")

# Archivo de clases (nos aseguramos de que busque 'class_names.json')
CLASS_PATH = MODEL_DIR / "class_names.json"

# Lista de formatos guardados para intentar cargar de forma dinámica (.keras y .h5)
MODEL_PATHS = [MODEL_DIR / "flores_net.keras", MODEL_DIR / "flores_net.h5"]
# ---------------------------------------------------

# Diccionario de traducción de las 5 categorías de Kaggle al español
LABELS_ES = {
    "daisy": "Margarita",
    "dandelion": "Diente de León",
    "rose": "Rosa",
    "sunflower": "Girasol",
    "tulip": "Tulipán",
    # Variantes con inicial mayúscula por si el generador las guardó así
    "Daisy": "Margarita",
    "Dandelion": "Diente de León",
    "Rose": "Rosa",
    "Sunflower": "Girasol",
    "Tulip": "Tulipán"
}

@st.cache_resource
def cargar_modelo():
    for path in MODEL_PATHS:
        if path.exists():
            try:
                return tf.keras.models.load_model(path, compile=False)
            except Exception:
                continue
    st.error(f"No se encontró el modelo. Verifique que los archivos estén dentro de la carpeta '{MODEL_DIR}'.")
    st.stop()

@st.cache_data
def cargar_clases():
    if CLASS_PATH.exists():
        try:
            with open(CLASS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Respaldo seguro si el archivo JSON no se lee correctamente
    return ["daisy", "dandelion", "rose", "sunflower", "tulip"]

def preparar_imagen(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    # Escalado idéntico al 1./255 del ImageDataGenerator de tu Colab
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

def predecir(img):
    preds = modelo.predict(preparar_imagen(img), verbose=0)[0]
    
    # Obtener los índices de las probabilidades más altas (máximo las 3 mejores)
    num_clases_mostrar = min(3, len(clases))
    top_indices = np.argsort(preds)[-num_clases_mostrar:][::-1]
    
    return [
        (LABELS_ES.get(clases[i], clases[i]), float(preds[i]) * 100)
        for i in top_indices
    ]

# Cargar el modelo y las clases al iniciar la app
modelo = cargar_modelo()
clases = cargar_clases()

# Componente para subir imágenes
archivo = st.file_uploader("Seleccione una fotografía de una flor", type=["jpg", "jpeg", "png"])

if archivo:
    imagen = Image.open(archivo)
    st.image(imagen, caption="Imagen cargada para análisis", use_container_width=True)

    with st.spinner("Analizando características de la flor..."):
        resultados = predecir(imagen)
        
    st.subheader("Resultado de la Clasificación")
    st.success(f"Predicción principal: **{resultados[0][0]}** ({resultados[0][1]:.2f}%)")

    st.write("Distribución de confianza (Top 3):")
    for clase, prob in resultados:
        st.write(f"- **{clase}**: {prob:.2f}%")
else:
    st.info("Cargue una imagen de una flor (Margarita, Diente de León, Rosa, Girasol o Tulipán) para iniciar la clasificación.")
