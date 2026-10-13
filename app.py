import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Predicción de Licitaciones",
    page_icon="🔮",
    layout="centered"
)

st.title("🔮 Sistema de Predicción de Licitaciones")

# ==========================================
# 1. CARGA DIRECTA DEL MODELO (SIN DRIVE)
# ==========================================
@st.cache_resource
def cargar_modelo():
    # Lee el archivo que ahora está directamente en GitHub
    return joblib.load("modelo_acumulativo.pkl")

try:
    models = cargar_modelo()
    st.success("✅ Modelo cargado exitosamente.")
except Exception as e:
    st.error(f"⚠️ Error al cargar el modelo. Detalle: {e}")
    st.stop()

target_cols = [
    'MEDIANA CON VALOR ABSOLUTO', 
    'MEDIA GEOMETRICA', 
    'MEDIA ARITMETICA BAJA', 
    'MENOR VALOR'
]

# (A partir de aquí, deja el resto de tu código exactamente igual)
