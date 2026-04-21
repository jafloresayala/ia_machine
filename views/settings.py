"""Settings — configuración de IA, API y preferencias."""
import streamlit as st
import requests

from theme import page_header
import config as app_config
from llm_engine import is_ollama_available


def render():
    page_header(
        eyebrow="Config",
        title="Settings",
        subtitle="Estado de los servicios, configuración de IA y acciones de mantenimiento.",
    )

    # -------- Estado de servicios --------
    with st.container(border=True):
        st.markdown("##### 🔌 Estado de servicios")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**PI Web API**")
            st.code(app_config.BASE_URL, language="text")
            st.caption(f"Plugin: `{app_config.PLUGIN_NAME}`")
            st.caption(f"Timeout: {app_config.API_TIMEOUT}s · SSL: {app_config.VERIFY_SSL}")
        with c2:
            st.markdown("**Ollama / LLM**")
            st.code(app_config.OLLAMA_BASE_URL, language="text")
            st.caption(f"Modelo: `{app_config.OLLAMA_MODEL}`")
            st.caption(f"Temperatura: {app_config.LLM_TEMPERATURE}")

        if st.button("🔄 Probar conexión con Ollama"):
            with st.spinner("Verificando…"):
                ok = is_ollama_available()
            if ok:
                st.success("✅ Ollama responde correctamente.")
                st.session_state.ollama_ok = True
            else:
                st.error("❌ No se pudo contactar con Ollama.")
                st.session_state.ollama_ok = False

    # -------- Root path --------
    with st.container(border=True):
        st.markdown("##### 🌳 Jerarquía PI")
        st.code(app_config.ROOT_PATH, language="text")
        df = st.session_state.get("machines_df")
        if df is not None and not df.empty:
            st.success(f"✅ {len(df)} máquinas descubiertas en {df['line_name'].nunique()} líneas.")
        else:
            st.warning("Sin catálogo cargado.")

    # -------- Cache --------
    with st.container(border=True):
        st.markdown("##### 🧹 Cache")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Recargar catálogo de máquinas", use_container_width=True):
                st.cache_data.clear()
                st.session_state.init_done = False
                st.rerun()
        with c2:
            if st.button("Limpiar insights generados", use_container_width=True):
                st.session_state.insights_cache = {}
                st.success("Cache de insights vaciado.")

    # -------- Info --------
    with st.container(border=True):
        st.markdown("##### ℹ️ Acerca de")
        st.markdown("""
        **Machine Intelligence Platform** · inspirado en Databricks Mosaic AI y Agent Bricks.

        - Backend de datos: PI Web API (`MBBP Data Fetch`)
        - Motor IA: Ollama + `deepseek-coder-v2`
        - Frontend: Streamlit + Altair
        - Análisis: pandas + numpy
        """)
