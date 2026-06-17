import streamlit as st

st.title("Generador de Informes PIE")

nombre = st.text_input("Nombre del estudiante")
curso = st.text_input("Curso")

if st.button("Generar Informe"):
    st.write(f"Informe generado para {nombre} de {curso}")