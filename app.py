import streamlit as st
import pandas as pd
import joblib
import gdown
import os

# ==========================================
# 0. CONFIGURACIÓN Y ENLACE DE DRIVE
# ==========================================
# ⚠️ Pega aquí el enlace público de tu archivo modelo_acumulativo.pkl en Google Drive
ENLACE_DRIVE = "https://drive.google.com/file/d/1aEJ85DDcA6V42VNbcqecfDLTzKC6YKB_/view?usp=sharing"
NOMBRE_ARCHIVO_LOCAL = "modelo_acumulativo.pkl"

st.set_page_config(
    page_title="Predicción de Licitaciones",
    page_icon="🔮",
    layout="centered"
)

st.title("🔮 Sistema de Predicción de Licitaciones")
st.markdown("Carga el modelo directamente desde Google Drive para calcular escenarios.")

# ==========================================
# 1. DESCARGA AUTOMÁTICA DESDE DRIVE (CORREGIDO)
# ==========================================
@st.cache_resource
def descargar_y_cargar_modelo(url_drive):
    if not os.path.exists(NOMBRE_ARCHIVO_LOCAL):
        with st.spinner("Descargando el modelo de IA más reciente desde Google Drive..."):
            # Se ajustó gdown.download eliminando el parámetro 'fuzzy' no compatible
            gdown.download(url=url_drive, output=NOMBRE_ARCHIVO_LOCAL, quiet=False)
    
    return joblib.load(NOMBRE_ARCHIVO_LOCAL)

try:
    models = descargar_y_cargar_modelo(ENLACE_DRIVE)
    st.success("✅ Modelo cargado exitosamente desde Google Drive.")
except Exception as e:
    st.error(f"⚠️ Error al conectar con Drive. Verifica el enlace de compartir. Detalle: {e}")
    st.stop()

target_cols = [
    'MEDIANA CON VALOR ABSOLUTO', 
    'MEDIA GEOMETRICA', 
    'MEDIA ARITMETICA BAJA', 
    'MENOR VALOR'
]

# ==========================================
# 2. PARÁMETROS E INTERFAZ
# ==========================================
st.sidebar.header("⚙️️ Parámetros de Entrada")

tipo_obra = st.sidebar.text_input("Tipo de Obra", value="MANTENIMIENTO")
universo = st.sidebar.number_input("Número de Oferentes (UNIVERSO)", min_value=1.0, max_value=200.0, value=35.0, step=1.0)
anio = st.sidebar.number_input("Año", min_value=2024, max_value=2030, value=2026, step=1)

# Botón para forzar actualización desde Drive si reentrenaste en Colab
if st.sidebar.button("🔄 Actualizar modelo desde Drive"):
    if os.path.exists(NOMBRE_ARCHIVO_LOCAL):
        os.remove(NOMBRE_ARCHIVO_LOCAL)
    st.cache_resource.clear()
    st.rerun()

# ==========================================
# 3. PREDICCIÓN Y RESULTADOS
# ==========================================
if st.button("🚀 Calcular Predicción", use_container_width=True):
    entrada = pd.DataFrame({
        'TIPO DE OBRA': [str(tipo_obra).strip().upper()],
        'UNIVERSO': [float(universo)],
        'AÑO': [float(anio)]
    })

    st.markdown("---")
    st.subheader(f"📊 Resultados de la Proyección ({tipo_obra.upper()})")
    st.caption(f"Configuración: **{int(universo)} Oferentes** | Año **{int(anio)}**")

    col1, col2 = st.columns(2)

    for i, col_name in enumerate(target_cols):
        prediccion = models[col_name].predict(entrada)[0]
        if i % 2 == 0:
            with col1:
                st.metric(label=col_name, value=f"{prediccion:.6f}")
        else:
            with col2:
                st.metric(label=col_name, value=f"{prediccion:.6f}")

    st.markdown("---")