"""Compare — comparación side-by-side entre máquinas (o mismo parámetro en varias)."""
from datetime import datetime

import streamlit as st
import pandas as pd
import altair as alt

from theme import page_header, COLORS
from api_client import fetch_tag_values
from machine_registry import discover_machines, get_machine_tags
from intent_parser import resolve_time_range
from dashboard_builder import infer_tag_mode
from analytics import numeric_summary
from chart_ai import chart_with_ai


def render():
    page_header(
        eyebrow="Intelligence · Compare",
        title="Compare",
        subtitle="Selecciona varias máquinas y compáralas lado a lado en el mismo período.",
    )

    df = discover_machines()
    if df is None or df.empty:
        st.warning("No hay máquinas disponibles.")
        return

    all_names = (df["machine_name"] + "  ·  " + df["line_name"]).tolist()

    c1, c2 = st.columns([3, 1])
    with c1:
        selected = st.multiselect(
            "Máquinas a comparar (2–4 recomendado)",
            options=all_names,
            default=st.session_state.get("compare_machines", all_names[: min(2, len(all_names))]),
        )
        st.session_state.compare_machines = selected
    with c2:
        window = st.selectbox(
            "Período",
            options=["today", "yesterday", "last_24h", "last_hour", "last_shift", "last_week"],
            format_func=lambda k: {
                "today":"Hoy","yesterday":"Ayer","last_24h":"Últimas 24h",
                "last_hour":"Última hora","last_shift":"Último turno","last_week":"Última semana",
            }[k],
        )

    if len(selected) < 2:
        st.info("Selecciona al menos 2 máquinas para comparar.")
        return

    from_dt, to_dt = resolve_time_range(window)
    rows_selected = [df.iloc[all_names.index(s)] for s in selected]

    # ---- Paso 1: cargar TODOS los tags de cada máquina (sin límite) para el dropdown ----
    machine_tags = {}   # machine_name -> {short_name: full_piPoint}
    for row in rows_selected:
        tags_df = get_machine_tags(row["machine_path"])
        if tags_df.empty:
            machine_tags[row["machine_name"]] = {}
            continue
        mapping = {}
        for _, tr in tags_df.iterrows():
            full = str(tr["piPoint"])
            short = full.split(".")[-1] if "." in full else full
            mapping[short] = full
        machine_tags[row["machine_name"]] = mapping

    # ---- Parámetros comunes (presentes en ≥2 máquinas) ----
    all_shorts = [set(m.keys()) for m in machine_tags.values()]
    common_sorted = sorted(set.intersection(*all_shorts)) if len(all_shorts) >= 2 else []

    # También incluir parámetros que coinciden en al menos 2 máquinas aunque no en todas
    if not common_sorted:
        from collections import Counter
        cnt = Counter()
        for mapping in machine_tags.values():
            cnt.update(mapping.keys())
        common_sorted = sorted(k for k, v in cnt.items() if v >= 2)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    if not common_sorted:
        st.info("No se encontraron parámetros con el mismo nombre entre las máquinas seleccionadas.")
        return

    sel_param = st.selectbox(
        f"Parámetro común a comparar ({len(common_sorted)} disponibles)",
        options=common_sorted,
    )

    # ---- Paso 2: cargar datos solo del parámetro seleccionado ----
    sources = []   # list of (machine_name, full_tag, df)
    with st.spinner(f"Cargando datos de «{sel_param}»…"):
        for row in rows_selected:
            mapping = machine_tags[row["machine_name"]]
            if sel_param not in mapping:
                continue
            full_tag = mapping[sel_param]
            raw = fetch_tag_values([full_tag], from_dt, to_dt)
            if raw.empty:
                continue
            sub = raw[raw["Tag_Name"] == full_tag].copy()
            if not sub.empty:
                sources.append((row["machine_name"], full_tag, sub))

    # ---- KPI comparativos (basados en el parámetro seleccionado) ----
    rows = []
    for name, mapping in machine_tags.items():
        n_params = len(mapping)
        rows.append({"Máquina": name, "Total parámetros": n_params})
    kpi_df = pd.DataFrame(rows)
    with st.container(border=True):
        st.markdown("##### 🧭 Resumen comparativo")
        st.dataframe(kpi_df, use_container_width=True, hide_index=True)

    # Sólo numéricos para overlay
    numeric_sources = [(m, tn, d) for (m, tn, d) in sources if infer_tag_mode(d) == "numeric"]

    if not numeric_sources:
        st.info("Ese parámetro no tiene datos numéricos en las máquinas seleccionadas para el período elegido.")
        return

    combined = []
    for m, tn, d in numeric_sources:
        sub = d.dropna(subset=["TimeStamp", "Value_Num"])[["TimeStamp", "Value_Num"]].copy()
        sub["Máquina"] = m
        combined.append(sub)
    plot_df = pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()

    if plot_df.empty:
        st.info("Sin datos suficientes para graficar.")
        return

    # ---- Gráfica overlay ----
    overlay = (
        alt.Chart(plot_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("TimeStamp:T", title="Tiempo"),
            y=alt.Y("Value_Num:Q", title=sel_param),
            color=alt.Color("Máquina:N", scale=alt.Scale(range=[
                COLORS["chart_1"], COLORS["chart_2"], COLORS["chart_3"],
                COLORS["chart_4"], COLORS["chart_5"], COLORS["chart_6"],
            ])),
            tooltip=["Máquina:N", "TimeStamp:T", alt.Tooltip("Value_Num:Q", format=".3f")],
        )
        .properties(height=380, title=f"Overlay: {sel_param}")
        .interactive()
    )
    with st.container(border=True):
        chart_with_ai(overlay, df=plot_df.rename(columns={"Value_Num": "Value_Num"}),
                      chart_id=f"compare_{sel_param}",
                      context={"param": sel_param, "type": "comparison_overlay",
                               "machines": [m for m, _, _ in numeric_sources]})

    # ---- Stats por máquina para el parámetro ----
    srows = []
    for m, tn, d in numeric_sources:
        s = numeric_summary(d)
        if not s:
            continue
        srows.append({
            "Máquina": m, "Último": round(s["last"], 3), "μ": round(s["mean"], 3),
            "σ": round(s["std"], 3), "CV%": round(s["cv_pct"], 1) if s["cv_pct"] == s["cv_pct"] else None,
            "Min": round(s["min"], 3), "Max": round(s["max"], 3),
            "Health": round(s["health"], 0),
        })
    if srows:
        with st.container(border=True):
            st.markdown(f"##### 📋 Estadísticas por máquina — {sel_param}")
            st.dataframe(pd.DataFrame(srows), use_container_width=True, hide_index=True)
