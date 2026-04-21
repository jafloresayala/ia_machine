"""Analytics Lab - analisis avanzado: outliers, correlacion, distribuciones, anomalias.

Lazy loading por pestana: los metadatos de tags se cargan sin limite al inicio,
pero los datos reales solo se descargan cuando el usuario selecciona parametros.
"""
import streamlit as st
import pandas as pd
import altair as alt

from theme import page_header, COLORS
from api_client import fetch_tag_values
from machine_registry import discover_machines, get_machine_tags
from intent_parser import resolve_time_range
from dashboard_builder import infer_tag_mode, _numeric_outlier_chart, _histogram_chart
from analytics import numeric_summary, correlation_matrix, detect_anomalies
from chart_ai import chart_with_ai


# ----------------------------------------------------------------
# Helper: obtener todos los tags de una maquina (sin limite, sin datos)
# ----------------------------------------------------------------
def _get_all_tags(machine_path: str):
    """Devuelve lista de piPoint completos y un dict short->full."""
    tags_df = get_machine_tags(machine_path)
    if tags_df.empty:
        return [], {}
    full_names = tags_df["piPoint"].astype(str).tolist()
    short_map = {}   # full_tag -> short_name
    for full in full_names:
        short = full.split(".")[-1] if "." in full else full
        short_map[full] = short
    return full_names, short_map


# ----------------------------------------------------------------
# Helper: fetch datos para los tags seleccionados
# ----------------------------------------------------------------
def _fetch_selected(tag_names: list, from_dt, to_dt) -> dict:
    """Retorna tag_data: {full_tag: DataFrame} solo para los tags dados."""
    if not tag_names:
        return {}
    with st.spinner(f"Cargando datos de {len(tag_names)} parametros..."):
        raw = fetch_tag_values(tag_names, from_dt, to_dt)
    td = {}
    for tn in tag_names:
        sub = raw[raw["Tag_Name"] == tn].copy()
        if not sub.empty:
            td[tn] = sub
    return td


# ----------------------------------------------------------------
# RENDER
# ----------------------------------------------------------------
def render():
    page_header(
        eyebrow="Intelligence - Analytics",
        title="Analytics Lab",
        subtitle="Laboratorio profundo: outliers, correlaciones, distribucion de valores y deteccion de anomalias.",
    )

    df = discover_machines()
    if df is None or df.empty:
        st.warning("No hay maquinas disponibles.")
        return

    # -- Selector de maquina y periodo (comun a todos los tabs) --
    c1, c2 = st.columns([3, 1])
    with c1:
        sel = st.selectbox(
            "Maquina",
            options=(df["machine_name"] + "  -  " + df["line_name"]).tolist(),
            index=0,
        )
    with c2:
        window = st.selectbox(
            "Periodo",
            options=["today", "yesterday", "last_24h", "last_hour", "last_shift", "last_week"],
            format_func=lambda k: {
                "today": "Hoy",
                "yesterday": "Ayer",
                "last_24h": "Ultimas 24h",
                "last_hour": "Ultima hora",
                "last_shift": "Ultimo turno",
                "last_week": "Ultima semana",
            }[k],
        )

    row = df.iloc[(df["machine_name"] + "  -  " + df["line_name"]).tolist().index(sel)]
    st.session_state.analytics_machine = row["machine_name"]

    # -- Cargar metadata de TODOS los tags (solo nombres, sin datos) --
    all_tags, short_map = _get_all_tags(row["machine_path"])
    if not all_tags:
        st.info("Esta maquina no tiene tags configurados.")
        return

    from_dt, to_dt = resolve_time_range(window)

    def fmt_tag(t):
        return short_map.get(t, t)

    # ============================================================
    t1, t2, t3, t4 = st.tabs(["Outliers", "Correlacion", "Distribuciones", "Anomalias"])

    # ============================================================
    # OUTLIERS
    # ============================================================
    with t1:
        st.markdown("Grafica con bandas upper/lower y porcentaje de outliers por parametro.")

        sel_outlier = st.multiselect(
            f"Parametros a analizar ({len(all_tags)} disponibles)",
            options=all_tags,
            default=all_tags[:min(3, len(all_tags))],
            format_func=fmt_tag,
            key="analytics_outlier_sel",
        )
        std_factor = st.slider("Sigma threshold", 1.0, 4.0, 2.0, 0.5, key="analytics_std")

        if not sel_outlier:
            st.info("Selecciona al menos un parametro para ver outliers.")
        else:
            td = _fetch_selected(sel_outlier, from_dt, to_dt)
            numeric_td = {n: d for n, d in td.items() if infer_tag_mode(d) == "numeric"}
            non_numeric = [n for n in td if n not in numeric_td]

            if non_numeric:
                st.caption(
                    f"Se omitieron {len(non_numeric)} parametros no numericos: "
                    + ", ".join(fmt_tag(n) for n in non_numeric)
                )

            if not numeric_td:
                st.info("Ninguno de los parametros seleccionados tiene datos numericos.")
            else:
                for tn, df_t in numeric_td.items():
                    short = fmt_tag(tn)
                    with st.container(border=True):
                        chart, stats = _numeric_outlier_chart(df_t, short, std_factor=std_factor)
                        if chart is None:
                            st.info(f"Sin datos suficientes en {short}.")
                            continue
                        chart_with_ai(chart, df=df_t, chart_id=f"analytics_outlier_{short}",
                                      context={"tag": short, "type": "outlier_analysis", "std_factor": std_factor})
                        mc = st.columns(4)
                        mc[0].metric("Sigma", f"{stats.get('std', 0):.3f}")
                        mc[1].metric("Upper", f"{stats.get('upper', 0):.3f}")
                        mc[2].metric("Lower", f"{stats.get('lower', 0):.3f}")
                        mc[3].metric("Outliers", f"{stats.get('outlier_count', 0)} ({stats.get('outlier_pct', 0):.1f}%)")

    # ============================================================
    # CORRELACION
    # ============================================================
    with t2:
        st.markdown(
            "Correlacion de Pearson entre parametros numericos, alineados por tiempo (resample 1 min). "
            "Solo se incluyen parametros con valores numericos."
        )

        sel_corr = st.multiselect(
            f"Parametros para correlacion ({len(all_tags)} disponibles - solo numericos seran usados)",
            options=all_tags,
            default=all_tags[:min(6, len(all_tags))],
            format_func=fmt_tag,
            key="analytics_corr_sel",
        )

        if len(sel_corr) < 2:
            st.info("Selecciona al menos 2 parametros para calcular la correlacion.")
        else:
            td = _fetch_selected(sel_corr, from_dt, to_dt)
            numeric_td = {n: d for n, d in td.items() if infer_tag_mode(d) == "numeric"}

            if len(numeric_td) < 2:
                st.warning(
                    f"Solo {len(numeric_td)} de los parametros seleccionados tienen datos numericos. "
                    "Se necesitan al menos 2 para calcular correlacion."
                )
            else:
                st.caption(f"Usando {len(numeric_td)} parametros numericos de {len(sel_corr)} seleccionados.")
                corr = correlation_matrix(numeric_td)

                if corr is None or corr.empty:
                    st.info("No hay suficientes datos coincidentes en el tiempo para calcular correlacion.")
                else:
                    corr_long = corr.reset_index().melt(id_vars="index", var_name="param_b", value_name="corr")
                    corr_long.rename(columns={"index": "param_a"}, inplace=True)
                    corr_long["param_a"] = corr_long["param_a"].apply(fmt_tag)
                    corr_long["param_b"] = corr_long["param_b"].apply(fmt_tag)

                    heat = (
                        alt.Chart(corr_long)
                        .mark_rect()
                        .encode(
                            x=alt.X("param_a:N", title=None),
                            y=alt.Y("param_b:N", title=None),
                            color=alt.Color(
                                "corr:Q",
                                scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                                title="rho",
                            ),
                            tooltip=[
                                alt.Tooltip("param_a:N", title="A"),
                                alt.Tooltip("param_b:N", title="B"),
                                alt.Tooltip("corr:Q", title="Pearson", format=".3f"),
                            ],
                        )
                        .properties(
                            height=max(260, 32 * len(corr)),
                            title="Matriz de correlacion",
                        )
                    )
                    text_layer = (
                        alt.Chart(corr_long)
                        .mark_text(baseline="middle", fontSize=10)
                        .encode(
                            x="param_a:N",
                            y="param_b:N",
                            text=alt.Text("corr:Q", format=".2f"),
                            color=alt.condition(
                                "abs(datum.corr) > 0.6",
                                alt.value("white"),
                                alt.value(COLORS["ink"]),
                            ),
                        )
                    )
                    chart_with_ai(heat + text_layer, df=None, chart_id="analytics_corr_heatmap",
                                  context={"type": "correlation_heatmap", "n_params": len(corr.columns)})

                    # Top pares correlacionados
                    pairs = []
                    names = list(corr.columns)
                    for i in range(len(names)):
                        for j in range(i + 1, len(names)):
                            r = corr.iloc[i, j]
                            if pd.isna(r):
                                continue
                            pairs.append({
                                "A": fmt_tag(names[i]),
                                "B": fmt_tag(names[j]),
                                "rho": round(r, 3),
                                "Fuerza": (
                                    "Fuerte" if abs(r) > 0.7
                                    else "Moderada" if abs(r) > 0.4
                                    else "Debil"
                                ),
                            })
                    if pairs:
                        pairs_df = (
                            pd.DataFrame(pairs)
                            .sort_values("rho", key=lambda s: s.abs(), ascending=False)
                        )
                        st.markdown("###### Pares mas correlacionados")
                        st.dataframe(pairs_df.head(20), use_container_width=True, hide_index=True)

    # ============================================================
    # DISTRIBUCIONES
    # ============================================================
    with t3:
        st.markdown("Histogramas de distribucion. Solo se muestran parametros con valores numericos.")

        sel_dist = st.multiselect(
            f"Parametros ({len(all_tags)} disponibles - solo numericos seran graficados)",
            options=all_tags,
            default=all_tags[:min(4, len(all_tags))],
            format_func=fmt_tag,
            key="analytics_dist_sel",
        )

        if not sel_dist:
            st.info("Selecciona al menos un parametro.")
        else:
            td = _fetch_selected(sel_dist, from_dt, to_dt)
            numeric_td = {n: d for n, d in td.items() if infer_tag_mode(d) == "numeric"}
            non_numeric = [n for n in td if n not in numeric_td]

            if non_numeric:
                st.caption(
                    f"Se omitieron {len(non_numeric)} parametros no numericos: "
                    + ", ".join(fmt_tag(n) for n in non_numeric)
                )

            if not numeric_td:
                st.info("Ninguno de los parametros seleccionados tiene datos numericos.")
            else:
                cols = st.columns(2)
                for i, (tn, df_t) in enumerate(numeric_td.items()):
                    short = fmt_tag(tn)
                    with cols[i % 2]:
                        with st.container(border=True):
                            st.markdown(f"**{short}**")
                            hist = _histogram_chart(df_t, short)
                            if hist is not None:
                                chart_with_ai(hist, df=df_t, chart_id=f"analytics_hist_{short}",
                                              context={"tag": short, "type": "histogram"})
                            s = numeric_summary(df_t)
                            if s:
                                m = st.columns(4)
                                m[0].metric("Media", f"{s['mean']:.2f}")
                                m[1].metric("Sigma", f"{s['std']:.2f}")
                                m[2].metric("IQR", f"{s['iqr']:.2f}")
                                m[3].metric("Rango", f"{s['range']:.2f}")

    # ============================================================
    # ANOMALIAS
    # ============================================================
    with t4:
        st.markdown("Puntos atipicos detectados por z-score. Solo parametros numericos.")

        sel_anom = st.multiselect(
            f"Parametros ({len(all_tags)} disponibles - solo numericos seran analizados)",
            options=all_tags,
            default=all_tags[:min(5, len(all_tags))],
            format_func=fmt_tag,
            key="analytics_anom_sel",
        )
        z_th = st.slider("Z-score threshold", 1.5, 4.0, 2.5, 0.25, key="analytics_z")

        if not sel_anom:
            st.info("Selecciona al menos un parametro.")
        else:
            td = _fetch_selected(sel_anom, from_dt, to_dt)
            numeric_td = {n: d for n, d in td.items() if infer_tag_mode(d) == "numeric"}

            if not numeric_td:
                st.info("Ninguno de los parametros seleccionados tiene datos numericos.")
            else:
                any_found = False
                for tn, df_t in numeric_td.items():
                    short = fmt_tag(tn)
                    anom = detect_anomalies(df_t, z_threshold=z_th)
                    if anom.empty:
                        continue
                    any_found = True
                    with st.container(border=True):
                        st.markdown(f"**{short}** - {len(anom)} puntos anomalos")
                        show = anom[["TimeStamp", "Value_Num", "z_score"]].copy()
                        show["z_score"] = show["z_score"].round(2)
                        st.dataframe(show.head(25), use_container_width=True, hide_index=True, height=280)
                if not any_found:
                    st.success("No se detectaron anomalias significativas en los parametros seleccionados.")
