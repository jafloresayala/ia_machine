"""Live Monitor — KPIs, health score y gráficas en tiempo real de una máquina."""
from datetime import datetime, timedelta, date, time

import streamlit as st
import pandas as pd

from theme import page_header, COLORS
from api_client import fetch_tag_values
from machine_registry import discover_machines, get_machine_tags, find_machine
from dashboard_builder import render_machine_dashboard, infer_tag_mode
from analytics import numeric_summary, categorical_summary, health_score_for_tags


def _pick_machine():
    df = discover_machines()
    if df is None or df.empty:
        st.warning("No hay máquinas disponibles.")
        return None, None

    cur_name = st.session_state.get("selected_machine_name")
    options = df["machine_name"] + "  ·  " + df["line_name"]
    default_idx = 0
    if cur_name:
        matches = df[df["machine_name"] == cur_name]
        if not matches.empty:
            default_idx = int(matches.index[0])

    sel = st.selectbox(
        "Selecciona una máquina",
        options=list(options),
        index=default_idx,
    )
    row_idx = list(options).index(sel)
    row = df.iloc[row_idx]
    st.session_state.selected_machine_name = row["machine_name"]
    st.session_state.selected_machine_path = row["machine_path"]
    return row["machine_name"], row["machine_path"]


def _pick_time_range() -> tuple[datetime, datetime, str]:
    """
    Devuelve (from_dt, to_dt, label).
    Modo Ventana: últimos N minutos desde ahora.
    Modo Rango:   fechas/horas exactas con precisión de segundos.
    """
    mode = st.radio(
        "Modo de tiempo",
        ["⏱ Ventana rápida", "📅 Rango exacto"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "⏱ Ventana rápida":
        window_min = st.select_slider(
            "Ventana",
            options=[5, 10, 15, 30, 60, 120, 240, 480, 1440],
            value=st.session_state.get("monitor_window", 60),
            format_func=lambda m: f"Últimos {m} min" if m < 60 else f"Últimas {m//60}h" if m % 60 == 0 else f"Últimas {m//60}h {m%60}m",
        )
        st.session_state.monitor_window = window_min
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(minutes=window_min)
        label = f"últimos {window_min} min" if window_min < 60 else f"últimas {window_min//60}h"
    else:
        now = datetime.now()
        default_from = now - timedelta(hours=1)

        st.markdown("""
        <div style='display:flex;gap:8px;align-items:center;margin-bottom:0.4rem;'>
          <span style='font-size:0.8rem;font-weight:600;color:#6366f1;letter-spacing:.05em;'>DESDE</span>
          <div style='flex:1;height:1px;background:linear-gradient(90deg,#6366f1,transparent)'></div>
        </div>""", unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns([3, 3, 1])
        with fc1:
            from_date = st.date_input("📅 Fecha", value=default_from.date(), key="mon_from_date")
        with fc2:
            from_time = st.time_input("🕐 Hora : Min", value=time(default_from.hour, default_from.minute, 0), key="mon_from_time", step=60)
        with fc3:
            from_sec = st.number_input("Seg", min_value=0, max_value=59, value=0, key="mon_from_sec")

        st.markdown("""
        <div style='display:flex;gap:8px;align-items:center;margin:0.5rem 0 0.4rem;'>
          <span style='font-size:0.8rem;font-weight:600;color:#06b6d4;letter-spacing:.05em;'>HASTA</span>
          <div style='flex:1;height:1px;background:linear-gradient(90deg,#06b6d4,transparent)'></div>
        </div>""", unsafe_allow_html=True)
        tc1, tc2, tc3 = st.columns([3, 3, 1])
        with tc1:
            to_date = st.date_input("📅 Fecha", value=now.date(), key="mon_to_date")
        with tc2:
            to_time = st.time_input("🕐 Hora : Min", value=time(now.hour, now.minute, 0), key="mon_to_time", step=60)
        with tc3:
            to_sec = st.number_input("Seg", min_value=0, max_value=59, value=0, key="mon_to_sec")

        from_dt = datetime.combine(from_date, from_time).replace(second=from_sec)
        to_dt   = datetime.combine(to_date,   to_time).replace(second=to_sec)

        if from_dt >= to_dt:
            st.error("⚠️ La fecha de inicio debe ser anterior a la fecha fin.")
            st.stop()

        delta = to_dt - from_dt
        total_sec = int(delta.total_seconds())
        total_min = total_sec // 60
        h, m = divmod(total_min, 60)
        duration_str = f"{h}h {m}m" if h else f"{m}m {total_sec % 60}s"
        st.markdown(
            f"<div style='margin-top:0.5rem;padding:0.45rem 0.75rem;border-radius:8px;"
            f"background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.2);"
            f"font-size:0.82rem;color:#6366f1;'>"
            f"⏱ <b>{from_dt.strftime('%d/%m/%Y %H:%M:%S')}</b>"
            f" &nbsp;→&nbsp; "
            f"<b>{to_dt.strftime('%d/%m/%Y %H:%M:%S')}</b>"
            f" &nbsp;·&nbsp; {duration_str}</div>",
            unsafe_allow_html=True,
        )
        label = f"{from_dt.strftime('%d/%m %H:%M:%S')} → {to_dt.strftime('%d/%m %H:%M:%S')} ({duration_str})"

    return from_dt, to_dt, label


def render():
    page_header(
        eyebrow="Data · Live",
        title="Live Monitor",
        subtitle="Estado actual, KPIs y tendencias de cualquier máquina en tiempo real.",
        right_html="<span class='pill pill-ok'>● LIVE</span>",
    )

    # ---- Máquina + rango de tiempo en un bloque compacto ----
    with st.container(border=True):
        name, path = _pick_machine()
        if not name:
            return
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        from_dt, to_dt, time_label = _pick_time_range()

    tags_df = get_machine_tags(path)
    if tags_df.empty:
        st.warning(f"La máquina **{name}** no tiene tags consultables.")
        return

    tag_names = tags_df["piPoint"].astype(str).tolist()
    with st.spinner(f"Consultando {len(tag_names)} tags…"):
        all_data = fetch_tag_values(tag_names, from_dt, to_dt)

    if all_data.empty:
        st.info("Sin datos en el rango seleccionado.")
        return

    tag_data = {}
    for tn in tag_names:
        sub = all_data[all_data["Tag_Name"] == tn].copy()
        if not sub.empty:
            tag_data[tn] = sub

    # ---- Salud general ----
    health = health_score_for_tags(tag_data)
    h = health.get("health")
    health_color = "ok" if h and h >= 75 else "warn" if h and h >= 50 else "err"

    kc = st.columns(5)
    with kc[0]: st.metric("Máquina", name)
    with kc[1]: st.metric("Rango", time_label[:30] + ("…" if len(time_label) > 30 else ""))
    with kc[2]: st.metric("Tags activos", len(tag_data))
    with kc[3]: st.metric("Puntos", f"{len(all_data):,}")
    with kc[4]:
        if h is not None:
            st.metric("Health Score", f"{h:.0f} / 100")
        else:
            st.metric("Health Score", "N/A")

    if h is not None:
        verdict = "Operación estable" if h >= 75 else "Revisar variabilidad" if h >= 50 else "Atención requerida"
        st.markdown(f"<span class='pill pill-{health_color}'>● {verdict}</span>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ---- Resumen de tags estilo Databricks ----
    with st.container(border=True):
        st.markdown("##### 📊 Parámetros — snapshot")
        rows = []
        for tn, df in tag_data.items():
            mode = infer_tag_mode(df)
            short = tn.split(".")[-1] if "." in tn else tn

            # Timestamp del último punto
            last_ts = "—"
            if "TimeStamp" in df.columns:
                ts_series = df["TimeStamp"].dropna()
                if not ts_series.empty:
                    ts_val = ts_series.iloc[-1]
                    try:
                        last_ts = pd.to_datetime(ts_val).strftime("%d/%m %H:%M:%S")
                    except Exception:
                        last_ts = str(ts_val)

            if mode == "numeric":
                s = numeric_summary(df)
                rows.append({
                    "Parámetro": short, "Tipo": "numeric",
                    "Timestamp": last_ts,
                    "Último valor": f"{s.get('last', float('nan')):.3f}" if s else "—",
                    "Mín":    f"{s.get('min', float('nan')):.3f}"  if s else "—",
                    "Máx":    f"{s.get('max', float('nan')):.3f}"  if s else "—",
                    "Promedio": f"{s.get('mean', float('nan')):.3f}" if s else "—",
                    "CV%":    f"{s.get('cv_pct', float('nan')):.1f}" if s else "—",
                    "Outliers": s.get("outliers_2std", 0) if s else 0,
                    "Health":  f"{s.get('health', 0):.0f}" if s else "—",
                })
            else:
                s = categorical_summary(df)
                rows.append({
                    "Parámetro": short, "Tipo": "state",
                    "Timestamp": last_ts,
                    "Último valor": s.get("last", "—") if s else "—",
                    "Mín": "—", "Máx": "—", "Promedio": "—", "CV%": "—",
                    "Outliers": "—",
                    "Health": f"{s.get('most_common_pct', 0):.0f}%" if s else "—",
                })
        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True, height=min(420, 80 + 36 * len(rows)))

    # ---- Dashboard completo ----
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        render_machine_dashboard(name, tag_data, time_range_label=time_label)

    # Botón refresh
    if st.button("🔄 Actualizar datos", type="primary", use_container_width=True):
        st.rerun()
