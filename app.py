"""
Machine Intelligence Platform - estilo Databricks Mosaic AI / Agent Bricks.

Shell principal: sidebar de navegacion + router a cada vista.
"""
import streamlit as st
from datetime import datetime

from theme import inject_global_css
from state import NAV_ITEMS, init_state, ensure_initialized
from config import OLLAMA_MODEL

# Vistas
from views import (
    home, agent, explorer, monitor, analytics, compare, insights, knowledge,
    settings as settings_view,
)


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Machine Intelligence Platform",
    page_icon="[MIP]",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# INIT
# =========================================================
inject_global_css()
init_state()
ensure_initialized()


# =========================================================
# SIDEBAR
# =========================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sb-brand">
          <div class="sb-logo">MIP</div>
          <div>
            <div class="sb-brand-name">Machine Intelligence</div>
            <div class="sb-brand-sub">Agent Kim</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        current = st.session_state.nav
        current_group = None
        for key, label, group in NAV_ITEMS:
            if group != current_group:
                st.markdown(f'<div class="sb-section">{group}</div>', unsafe_allow_html=True)
                current_group = group
            btn_type = "primary" if current == key else "secondary"
            if st.button(label, key=f"nav_{key}", type=btn_type, use_container_width=True):
                if current != key:
                    st.session_state.nav = key
                    st.rerun()

        st.markdown('<div class="sb-section">System</div>', unsafe_allow_html=True)
        ai_ok = st.session_state.get("ollama_ok", False)
        machines_loaded = st.session_state.get("machines_loaded", False)
        df = st.session_state.get("machines_df")
        n_machines = len(df) if df is not None and not df.empty else 0

        ai_dot = "dot-ok" if ai_ok else "dot-err"
        ai_text = f"IA - {OLLAMA_MODEL}" if ai_ok else "IA offline"
        st.markdown(f"""
        <div class="sb-status">
          <span class="dot {ai_dot}"></span>
          <span style="font-size:0.78rem;">{ai_text}</span>
        </div>
        """, unsafe_allow_html=True)

        pi_dot = "dot-ok" if machines_loaded else "dot-warn"
        st.markdown(f"""
        <div class="sb-status">
          <span class="dot {pi_dot}"></span>
          <span style="font-size:0.78rem;">PI - {n_machines} maquinas</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sb-status">
          <span class="dot dot-ok"></span>
          <span style="font-size:0.78rem;">{datetime.now().strftime("%H:%M - %d %b")}</span>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# ROUTER
# =========================================================
VIEWS = {
    "home":      home.render,
    "agent":     agent.render,
    "explorer":  explorer.render,
    "monitor":   monitor.render,
    "analytics": analytics.render,
    "compare":   compare.render,
    "insights":  insights.render,
    "knowledge": knowledge.render,
    "settings":  settings_view.render,
}


def main():
    render_sidebar()
    view_fn = VIEWS.get(st.session_state.nav, home.render)
    try:
        view_fn()
    except Exception as e:
        st.error(f"Error en la vista: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
else:
    main()
