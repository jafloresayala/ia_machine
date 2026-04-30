"""Home — dashboard general / Agent Kim."""
import streamlit as st
import pandas as pd
from datetime import datetime

from theme import page_header, COLORS
from state import goto
from config import OLLAMA_MODEL, ROOT_PATH
from machine_registry import discover_machines


def render():
    df = st.session_state.get("machines_df")
    if df is None:
        df = discover_machines()

    n_machines = 0 if df is None or df.empty else len(df)
    n_lines = 0 if df is None or df.empty else df["line_name"].nunique()
    ai_ok = st.session_state.get("ollama_ok", False)

    # ------- Hero ---------
    st.markdown(f"""
    <div class="hero">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:0.8rem;">
        <div style="width:46px;height:46px;border-radius:12px;background:linear-gradient(135deg,#06b6d4,#a78bfa);display:flex;align-items:center;justify-content:center;font-size:22px;">⚡</div>
        <div>
          <div style="font-size:0.7rem;letter-spacing:0.16em;text-transform:uppercase;color:#a5b4fc;font-weight:700;">Machine Intelligence Platform</div>
          <h1>Agent Kim para tu piso de planta</h1>
        </div>
      </div>
      <p>Explora, monitorea y entiende cualquier parámetro de cualquier máquina SMT con ayuda de un agente de IA. Combina datos en tiempo real de PI Web API con análisis inteligente para operaciones de clase mundial.</p>
      <div class="hero-chips">
        <span class="hero-chip">🏭 {n_machines} máquinas</span>
        <span class="hero-chip">🔀 {n_lines} líneas</span>
        <span class="hero-chip">🧠 {OLLAMA_MODEL if ai_ok else 'IA offline'}</span>
        <span class="hero-chip">🕒 {datetime.now().strftime("%H:%M")}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ------- Métricas clave ---------
    cols = st.columns(4)
    with cols[0]: st.metric("Máquinas conectadas", f"{n_machines}")
    with cols[1]: st.metric("Líneas de producción", f"{n_lines}")
    with cols[2]: st.metric("Motor IA", "Online ✓" if ai_ok else "Offline ✗")
    with cols[3]: st.metric("Última sincronización", datetime.now().strftime("%H:%M"))

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ------- Capabilities grid ---------
    st.markdown("##### Capabilities")
    cap_cols = st.columns(3)
    capabilities = [
        ("🤖", "Agente conversacional",
         "Pregunta en lenguaje natural y recibe dashboards, análisis y recomendaciones.",
         "agent"),
        ("🗂️", "Explorer",
         "Navega por la jerarquía completa de líneas, máquinas y parámetros.",
         "explorer"),
        ("📡", "Live Monitor",
         "Monitoreo en tiempo real con KPIs, alertas y health scores.",
         "monitor"),
        ("🎮", "Playground",
         "Outliers, correlaciones, distribuciones y detección de anomalías.",
         "analytics"),
        ("⚖️", "Compare",
         "Comparación side-by-side entre máquinas, líneas o períodos.",
         "compare"),
    ]
    for i, (icon, title, desc, target) in enumerate(capabilities):
        with cap_cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.4rem;">
                  <div style="width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,rgba(79,70,229,.12),rgba(6,182,212,.12));display:flex;align-items:center;justify-content:center;font-size:18px;">{icon}</div>
                  <div style="font-weight:700;font-size:0.98rem;color:{COLORS['ink']};">{title}</div>
                </div>
                <div style="color:{COLORS['ink_muted']};font-size:0.87rem;line-height:1.45;min-height:60px;">{desc}</div>
                """, unsafe_allow_html=True)
                if st.button("Abrir →", key=f"home_go_{target}", use_container_width=True):
                    goto(target)

    st.markdown(f"""
    <div class="app-footer">
      Source: PI Web API · <span class="mono">{ROOT_PATH.split(chr(92))[-1]}</span> · Powered by Ollama
    </div>
    """, unsafe_allow_html=True)
