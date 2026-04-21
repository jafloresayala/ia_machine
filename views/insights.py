"""AI Insights — resúmenes y hallazgos generados automáticamente por el LLM."""
from datetime import datetime

import streamlit as st
import pandas as pd

from theme import page_header, COLORS
from api_client import fetch_tag_values
from machine_registry import discover_machines, get_machine_tags
from intent_parser import resolve_time_range
from llm_engine import chat_completion, OLLAMA_MODEL
from dashboard_builder import infer_tag_mode, build_data_summary_text
from analytics import numeric_summary, detect_anomalies, health_score_for_tags


INSIGHTS_SYSTEM_PROMPT = """Eres un ingeniero senior de manufactura SMT que revisa datos operativos.
Tu tarea es producir INSIGHTS accionables sobre una máquina, en español, con formato Markdown.

Estructura tu respuesta con estas secciones:

### 🎯 Estado general
Una evaluación corta (2-3 líneas) del estado actual.

### ✅ Fortalezas
- Lista de parámetros que están bien
- Usa cifras concretas

### ⚠️ Riesgos
- Parámetros con alta variabilidad, outliers, o drift
- Usa cifras concretas

### 🔧 Recomendaciones
- Acciones concretas, numeradas
- Prioriza por impacto

Sé preciso, honesto y profesional. Máximo 300 palabras. NO inventes datos."""


def _generate_insight(machine_name: str, tag_data: dict) -> str:
    if not tag_data:
        return "_Sin datos suficientes._"
    data_summary = build_data_summary_text(tag_data)
    # Enriquecer con stats
    lines = []
    for tn, d in tag_data.items():
        if infer_tag_mode(d) == "numeric":
            s = numeric_summary(d)
            if s:
                short = tn.split(".")[-1] if "." in tn else tn
                lines.append(
                    f"- {short}: μ={s['mean']:.3f} σ={s['std']:.3f} "
                    f"CV%={s['cv_pct']:.1f} outliers={s['outliers_2std']} health={s['health']:.0f}"
                )
    extra = "\n".join(lines)
    user_msg = (
        f"MÁQUINA: {machine_name}\n\n"
        f"RESUMEN BÁSICO:\n{data_summary}\n\n"
        f"ESTADÍSTICAS DETALLADAS:\n{extra}\n\n"
        f"Genera los insights ingenieriles."
    )
    try:
        return chat_completion(INSIGHTS_SYSTEM_PROMPT, user_msg, temperature=0.3)
    except Exception as e:
        return f"_Error generando insight: {e}_"


def render():
    ai_ok = st.session_state.get("ollama_ok", False)
    page_header(
        eyebrow="Intelligence · Insights",
        title="AI Insights",
        subtitle="El agente analiza una máquina y produce un reporte ejecutivo con riesgos y recomendaciones.",
        right_html=(
            f"<span class='pill pill-ok'>● {OLLAMA_MODEL}</span>" if ai_ok else
            "<span class='pill pill-err'>● IA offline</span>"
        ),
    )

    if not ai_ok:
        st.warning("El motor de IA (Ollama) no está disponible. Inicia Ollama y recarga la aplicación.")
        return

    df = discover_machines()
    if df is None or df.empty:
        st.warning("No hay máquinas disponibles.")
        return

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        sel = st.selectbox(
            "Máquina", options=(df["machine_name"] + "  ·  " + df["line_name"]).tolist(),
        )
    with c2:
        window = st.selectbox(
            "Período",
            options=["today", "yesterday", "last_24h", "last_shift"],
            format_func=lambda k: {"today":"Hoy","yesterday":"Ayer","last_24h":"24h","last_shift":"Turno"}[k],
        )
    with c3:
        st.write("")
        regenerate = st.button("🔁 Regenerar", use_container_width=True)

    row = df.iloc[(df["machine_name"] + "  ·  " + df["line_name"]).tolist().index(sel)]
    cache_key = f"{row['machine_name']}|{window}"

    tags_df = get_machine_tags(row["machine_path"])
    if tags_df.empty:
        st.warning("La máquina no tiene tags consultables.")
        return

    from_dt, to_dt = resolve_time_range(window)
    tag_names = tags_df["piPoint"].astype(str).tolist()
    with st.spinner("Cargando datos…"):
        all_data = fetch_tag_values(tag_names, from_dt, to_dt)

    tag_data = {}
    for tn in tag_names:
        sub = all_data[all_data["Tag_Name"] == tn].copy()
        if not sub.empty:
            tag_data[tn] = sub

    if not tag_data:
        st.info("Sin datos para generar insight.")
        return

    # Health + top anomalies
    health = health_score_for_tags(tag_data)
    h = health.get("health")
    kc = st.columns(4)
    with kc[0]: st.metric("Parámetros", len(tag_data))
    with kc[1]: st.metric("Puntos", f"{len(all_data):,}")
    with kc[2]: st.metric("Health Score", f"{h:.0f}" if h is not None else "—")
    # Contar anomalías
    total_anom = 0
    for tn, d in tag_data.items():
        if infer_tag_mode(d) == "numeric":
            a = detect_anomalies(d, z_threshold=2.5)
            total_anom += len(a)
    with kc[3]: st.metric("Anomalías (z>2.5σ)", total_anom)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # --- Insight del LLM ---
    cache = st.session_state.get("insights_cache", {})
    if regenerate or cache_key not in cache:
        with st.spinner("🧠 Generando insights con IA…"):
            insight = _generate_insight(row["machine_name"], tag_data)
        cache[cache_key] = insight
        st.session_state.insights_cache = cache
    else:
        insight = cache[cache_key]

    with st.container(border=True):
        st.markdown("#### 💡 Reporte de insights")
        st.markdown(insight)

    # --- Top anomalías detalle ---
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    with st.expander("🔍 Ver detalle de anomalías detectadas"):
        found_any = False
        for tn, d in tag_data.items():
            if infer_tag_mode(d) != "numeric":
                continue
            a = detect_anomalies(d, z_threshold=2.5)
            if a.empty:
                continue
            found_any = True
            short = tn.split(".")[-1] if "." in tn else tn
            st.markdown(f"**{short}** — {len(a)} puntos")
            show = a[["TimeStamp", "Value_Num", "z_score"]].copy()
            show["z_score"] = show["z_score"].round(2)
            st.dataframe(show.head(10), use_container_width=True, hide_index=True)
        if not found_any:
            st.success("Sin anomalías significativas.")
