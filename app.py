import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Clasificador de Frutas IA", layout="centered")

# --- ENCABEZADO OBLIGATORIO (INFORMACIÓN DEL ESTUDIANTE) ---
st.title("Clasificación de Frutas con Inteligencia Artificial (Fruits-360)")
st.markdown("### **Asignatura:** IS-701 - Inteligencia Artificial (Campus Comayagua)")
st.markdown("### **Estudiante:** Edwin Eduardo Guzmán Ramos")
st.markdown("### **Número de Cuenta:** 20211930058")
st.write("Suba una imagen para clasificarla mediante el modelo MobileNetV2 entrenado a 6 épocas.")
st.write("---") 

IMG_SIZE = (224, 224)

# --- CONFIGURACIÓN DE RUTAS DE ACUERDO A TU GITHUB ---
# Apuntamos exactamente a la carpeta que tienes en tu repositorio
MODEL_DIR = Path("modelo_frutas")

# Archivo de clases dentro de tu carpeta (con la extensión en el script por seguridad)
CLASS_PATH = MODEL_DIR / "fruits_class_names.json"

# Modelos dentro de tu carpeta
MODEL_PATHS = [
    MODEL_DIR / "fruits_mobile_net.keras", 
    MODEL_DIR / "fruits_mobile_net.h5"
]
# ---------------------------------------------------

# Diccionario de traducción de las 10 categorías reales de Kaggle al español
LABELS_ES = {
    "Apple 10": "Manzana Tipo 10",
    "Apple 11": "Manzana Tipo 11",
    "Apple 12": "Manzana Tipo 12",
    "Banana 1": "Banano (Maduro)",
    "Banana 3": "Banano (Variante)",
    "Blueberry 1": "Arándano Azul",
    "Onion Red 1": "Cebolla Morada",
    "Onion White 1": "Cebolla Blanca",
    "Orange 1": "Naranja Común",
    "Orange 2": "Naranja Variante"
}

@st.cache_resource
def cargar_modelo():
    for path in MODEL_PATHS:
        if path.exists():
            try:
                return tf.keras.models.load_model(path, compile=False)
            except Exception:
                continue
    st.error(f"No se encontró el modelo de frutas. Verifique que los archivos estén dentro de la carpeta '{MODEL_DIR}'.")
    st.stop()

@st.cache_data
def cargar_clases():
    # Intentar leer desde el archivo JSON si existe con extensión
    if CLASS_PATH.exists():
        try:
            with open(CLASS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
            
    # Intento de lectura de respaldo sin extensión (por la captura de Windows)
    PATH_SIN_EXT = MODEL_DIR / "fruits_class_names"
    if PATH_SIN_EXT.exists():
        try:
            with open(PATH_SIN_EXT, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # LISTA DE RESPALDO DEFINITIVA (Mantiene el orden exacto de tus neuronas [0-9])
    return [
        "Apple 10", 
        "Apple 11", 
        "Apple 12", 
        "Banana 1", 
        "Banana 3", 
        "Blueberry 1", 
        "Onion Red 1", 
        "Onion White 1", 
        "Orange 1", 
        "Orange 2"
    ]

def preparar_imagen(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    # Pipeline de normalización idéntico a tu entrenamiento de Colab
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

def predecir(img):
    preds = modelo.predict(preparar_imagen(img), verbose=0)[0]
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
archivo = st.file_uploader("Seleccione una fotografía de una fruta o vegetal", type=["jpg", "jpeg", "png"])

if archivo:
    imagen = Image.open(archivo)
    st.image(imagen, caption="Imagen cargada para análisis", use_container_width=True)

    with st.spinner("Analizando características geométricas y texturas..."):
        resultados = predecir(imagen)
        
    st.subheader("Resultado de la Clasificación")
    st.success(f"Predicción principal: **{resultados[0][0]}** ({resultados[0][1]:.2f}%)")

    st.write("Distribución de confianza (Top probabilidades):")
    for clase, prob in resultados:
        st.write(f"- **{clase}**: {prob:.2f}%")
else:
    st.info("Cargue una imagen de las categorías válidas: Manzana (10, 11, 12), Banano (1, 3), Arándano, Cebolla (Morada, Blanca) o Naranja (1, 2) para iniciar la clasificación.")
