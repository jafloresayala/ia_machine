"""Estado global de la aplicación y utilidades compartidas entre vistas."""
import streamlit as st
from datetime import datetime, timedelta

from llm_engine import is_ollama_available
from machine_registry import discover_machines


NAV_ITEMS = [
    ("home",       "🏠 Home",            "Overview"),
    ("agent",      "🤖 Agent",           "AI"),
    ("explorer",   "🗂️ Explorer",         "Data"),
    ("monitor",    "📡 Live Monitor",    "Data"),
    ("analytics",  "🔬 Analytics Lab",   "Intelligence"),
    ("compare",    "⚖️ Compare",          "Intelligence"),
    ("insights",   "💡 AI Insights",     "Intelligence"),
    ("knowledge",  "📚 Knowledge",       "Library"),
    ("settings",   "⚙️ Settings",         "Config"),
]


def init_state():
    """Inicializa claves de session_state usadas globalmente."""
    defaults = {
        "nav":                "home",
        "ollama_ok":          None,       # lazy-check
        "machines_df":        None,
        "machines_loaded":    False,
        "init_done":          False,

        # Agent
        "chat_history":       [],
        "pending_intent":     None,
        "ui_time_range":      "today",
        "ui_custom_from":     datetime.now().date(),
        "ui_custom_to":       datetime.now().date(),
        "opt_dashboard":      True,
        "opt_analysis":       True,
        "opt_excel":          False,

        # Monitor / Explorer
        "selected_machine_path":  None,
        "selected_machine_name":  None,
        "monitor_refresh":    60,  # seconds
        "explorer_query":     "",

        # Compare
        "compare_machines":   [],

        # Analytics
        "analytics_machine":  None,
        "analytics_range":    "today",

        # Insights
        "insights_cache":     {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def ensure_initialized():
    """Descubre máquinas y verifica Ollama una sola vez."""
    if st.session_state.init_done:
        return

    with st.spinner("Conectando con el servicio de IA..."):
        try:
            st.session_state.ollama_ok = is_ollama_available()
        except Exception:
            st.session_state.ollama_ok = False

    with st.spinner("Descubriendo máquinas PI..."):
        try:
            df = discover_machines()
            st.session_state.machines_df = df
            st.session_state.machines_loaded = not df.empty
        except Exception:
            st.session_state.machines_df = None
            st.session_state.machines_loaded = False

    st.session_state.init_done = True


def goto(view_key: str):
    """Cambia de vista y fuerza rerun."""
    st.session_state.nav = view_key
    st.rerun()
