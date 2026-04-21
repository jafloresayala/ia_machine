"""
Construye dashboards automáticos a partir de los datos obtenidos.
Genera componentes Streamlit directamente.
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

from chart_ai import chart_with_ai


# ----------------------------------------------------------
# Detección de tipo de datos
# ----------------------------------------------------------
def infer_tag_mode(df: pd.DataFrame) -> str:
    if df.empty:
        return "unknown"

    tag_type = ""
    if "Tag_Type" in df.columns and df["Tag_Type"].notna().any():
        tag_type = str(df["Tag_Type"].dropna().iloc[0]).lower()

    if any(x in tag_type for x in ("float", "double", "int", "decimal", "numeric")):
        return "numeric"
    if "bool" in tag_type or tag_type == "boolean":
        return "boolean"
    if "string" in tag_type or "char" in tag_type:
        return "categorical"

    if "Value_Num" in df.columns:
        valid_ratio = df["Value_Num"].notna().mean() if len(df) > 0 else 0
        if valid_ratio >= 0.8:
            return "numeric"

    unique_vals = set(df["Value_Str"].dropna().astype(str).str.lower().unique())
    if unique_vals and unique_vals.issubset({"true", "false", "0", "1", "yes", "no", "on", "off"}):
        return "boolean"

    return "categorical"


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def _safe_metric(value, decimals=2):
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    try:
        return f"{value:,.{decimals}f}"
    except Exception:
        return str(value)


def _safe_ts(ts):
    if ts is None:
        return "N/A"
    try:
        if pd.isna(ts):
            return "N/A"
    except Exception:
        pass
    try:
        return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


# ----------------------------------------------------------
# Gráfico numérico
# ----------------------------------------------------------
def _numeric_chart(df: pd.DataFrame, tag_name: str) -> alt.Chart:
    base = alt.Chart(df).encode(
        x=alt.X("TimeStamp:T", title="Tiempo"),
    )

    line = base.mark_line(color="#3b82f6", strokeWidth=2).encode(
        y=alt.Y("Value_Num:Q", title="Valor"),
        tooltip=[
            alt.Tooltip("TimeStamp:T", title="Hora"),
            alt.Tooltip("Value_Num:Q", title="Valor", format=",.3f"),
        ],
    )

    points = base.mark_circle(color="#3b82f6", size=30, opacity=0.5).encode(
        y=alt.Y("Value_Num:Q"),
        tooltip=[
            alt.Tooltip("TimeStamp:T", title="Hora"),
            alt.Tooltip("Value_Num:Q", title="Valor", format=",.3f"),
        ],
    )

    return alt.layer(line, points).properties(height=300, title=tag_name).interactive()


# ----------------------------------------------------------
# Gráficos categóricos / booleanos — lógica adaptativa
# ----------------------------------------------------------
def _categorical_chart(df: pd.DataFrame, tag_name: str) -> alt.Chart:
    """Legacy scatter — kept as fallback."""
    return (
        alt.Chart(df)
        .mark_circle(size=80)
        .encode(
            x=alt.X("TimeStamp:T", title="Tiempo"),
            y=alt.Y("Value_Str:N", title="Estado"),
            color=alt.Color("Value_Str:N", title="Valor"),
            tooltip=["TimeStamp:T", "Value_Str:N"],
        )
        .properties(height=250, title=tag_name)
        .interactive()
    )


def _smart_categorical_chart(df: pd.DataFrame, tag_name: str, mode: str) -> alt.Chart | None:
    """
    Elige automaticamente el mejor grafico segun el tipo de dato.
    Intenta extraer valores de Value_Str -> Value_Raw -> Value_Num en ese orden.
    """
    _NULLS = {"nan", "none", "null", "", "na", "n/a"}

    def _clean_series(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().where(
            ~s.astype(str).str.strip().str.lower().isin(_NULLS)
        )

    plot_df = df.dropna(subset=["TimeStamp"]).copy()

    # Intentar obtener una columna de valores en orden de prioridad
    val_col = None
    for col in ("Value_Str", "Value_Raw"):
        if col in plot_df.columns:
            cleaned = _clean_series(plot_df[col])
            if cleaned.notna().sum() > 0:
                plot_df["Value_Str"] = cleaned
                val_col = "Value_Str"
                break

    # Ultimo recurso: Value_Num convertido a string
    if val_col is None and "Value_Num" in plot_df.columns:
        num_clean = plot_df["Value_Num"].dropna()
        if not num_clean.empty:
            plot_df["Value_Str"] = plot_df["Value_Num"].apply(
                lambda v: str(int(v)) if pd.notna(v) and float(v) == int(float(v))
                else (f"{v:.4g}" if pd.notna(v) else pd.NA)
            )
            val_col = "Value_Str"

    if val_col is None:
        return None

    # Descartar filas sin valor
    plot_df = plot_df.dropna(subset=["Value_Str"]).copy()
    if plot_df.empty:
        return None

    unique_vals = plot_df["Value_Str"].unique()
    n_unique = len(unique_vals)

    # --- Boolean: step-area chart ---
    bool_map = {"true": True, "false": False, "1": True, "0": False,
                "on": True, "off": False, "yes": True, "no": False}
    is_bool = mode == "boolean" or (
        n_unique <= 2 and all(v.lower() in bool_map for v in unique_vals)
    )

    if is_bool:
        plot_df["State_Num"] = plot_df["Value_Str"].str.lower().map(
            lambda v: 1.0 if bool_map.get(v, False) else 0.0
        )
        chart = (
            alt.Chart(plot_df)
            .mark_area(interpolate="step-after", opacity=0.6)
            .encode(
                x=alt.X("TimeStamp:T", title="Tiempo"),
                y=alt.Y("State_Num:Q", title="Estado",
                        scale=alt.Scale(domain=[-0.1, 1.3]),
                        axis=alt.Axis(values=[0, 1],
                                      labelExpr="datum.value === 1 ? 'ON' : 'OFF'")),
                color=alt.Color("Value_Str:N", title="Estado",
                                scale=alt.Scale(range=["#ef4444", "#22c55e"])),
                tooltip=[alt.Tooltip("TimeStamp:T", title="Hora"),
                         alt.Tooltip("Value_Str:N", title="Estado")],
            )
            .properties(height=280, title=f"{tag_name} — ON/OFF")
            .interactive()
        )
        return chart

    # --- Pocos estados (<=12): gantt horizontal ---
    if n_unique <= 12:
        plot_df = plot_df.sort_values("TimeStamp").reset_index(drop=True)
        if len(plot_df) == 1:
            # Solo un punto: mostrar como scatter simple
            return (
                alt.Chart(plot_df)
                .mark_point(size=150, filled=True)
                .encode(
                    x=alt.X("TimeStamp:T", title="Tiempo"),
                    color=alt.Color("Value_Str:N", title="Estado"),
                    tooltip=[alt.Tooltip("TimeStamp:T"), alt.Tooltip("Value_Str:N", title="Estado")],
                )
                .properties(height=200, title=tag_name)
                .interactive()
            )
        # Construir intervalos
        intervals = []
        for i in range(len(plot_df)):
            start = plot_df.at[i, "TimeStamp"]
            if i + 1 < len(plot_df):
                end = plot_df.at[i + 1, "TimeStamp"]
            else:
                # Último punto: extender 1 minuto o la mediana del intervalo
                typical = (plot_df["TimeStamp"].diff().dropna().median()
                           if len(plot_df) > 1 else pd.Timedelta(minutes=1))
                end = start + (typical if pd.notna(typical) else pd.Timedelta(minutes=1))
            intervals.append({"state": plot_df.at[i, "Value_Str"], "start": start, "end": end})

        gantt_df = pd.DataFrame(intervals)
        palette = ["#3b82f6","#22c55e","#f59e0b","#ef4444","#8b5cf6",
                   "#06b6d4","#f97316","#ec4899","#84cc16","#a78bfa","#fb923c","#38bdf8"]
        chart = (
            alt.Chart(gantt_df)
            .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
            .encode(
                x=alt.X("start:T", title="Inicio"),
                x2=alt.X2("end:T"),
                y=alt.Y("state:N", title="Estado", sort="-x"),
                color=alt.Color("state:N", title="Estado",
                                scale=alt.Scale(range=palette[:n_unique])),
                tooltip=[
                    alt.Tooltip("state:N", title="Estado"),
                    alt.Tooltip("start:T", title="Inicio"),
                    alt.Tooltip("end:T",   title="Fin"),
                ],
            )
            .properties(height=max(260, n_unique * 55),
                        title=f"{tag_name} — Línea de estados")
            .interactive()
        )
        return chart

    # --- Muchos estados: frecuencia horizontal ---
    freq = (
        plot_df["Value_Str"].value_counts().reset_index()
        .rename(columns={"Value_Str": "Estado", "count": "Veces"})
    )
    chart = (
        alt.Chart(freq)
        .mark_bar(color="#3b82f6", cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("Veces:Q", title="Ocurrencias"),
            y=alt.Y("Estado:N", sort="-x", title="Valor"),
            tooltip=["Estado:N", "Veces:Q"],
        )
        .properties(height=max(300, min(600, n_unique * 34 + 60)),
                    title=f"{tag_name} — Frecuencia de valores")
    )
    return chart


# ----------------------------------------------------------
# Gráfico numérico con límites de outliers (std dev)
# ----------------------------------------------------------
def _numeric_outlier_chart(df: pd.DataFrame, tag_name: str, std_factor: float = 2.0) -> tuple[alt.Chart, dict]:
    """
    Gráfico de línea + puntos con bandas upper/lower basadas en desviación estándar.
    Puntos fuera de rango se marcan en rojo.
    Devuelve (chart, stats_dict).
    """
    plot_df = df.dropna(subset=["TimeStamp", "Value_Num"]).copy()
    if plot_df.empty:
        return None, {}

    mean_val = plot_df["Value_Num"].mean()
    std_val = plot_df["Value_Num"].std(ddof=0)
    upper = mean_val + std_factor * std_val
    lower = mean_val - std_factor * std_val

    plot_df["Is_Outlier"] = (plot_df["Value_Num"] < lower) | (plot_df["Value_Num"] > upper)
    plot_df["Upper_Limit"] = upper
    plot_df["Lower_Limit"] = lower

    normal_df = plot_df[~plot_df["Is_Outlier"]]
    outlier_df = plot_df[plot_df["Is_Outlier"]]

    base_x = alt.X("TimeStamp:T", title="Tiempo")

    line = alt.Chart(plot_df).mark_line(color="#3b82f6", strokeWidth=2).encode(
        x=base_x,
        y=alt.Y("Value_Num:Q", title="Valor"),
        tooltip=[
            alt.Tooltip("TimeStamp:T", title="Hora"),
            alt.Tooltip("Value_Num:Q", title="Valor", format=",.3f"),
        ],
    )

    normal_pts = alt.Chart(normal_df).mark_circle(color="#3b82f6", size=35, opacity=0.6).encode(
        x=base_x, y="Value_Num:Q",
        tooltip=[alt.Tooltip("TimeStamp:T"), alt.Tooltip("Value_Num:Q", format=",.3f")],
    )

    layers = [line, normal_pts]

    if not outlier_df.empty:
        outlier_pts = alt.Chart(outlier_df).mark_circle(
            color="#ef4444", size=110, opacity=0.95, stroke="white", strokeWidth=1.2,
        ).encode(
            x=base_x, y="Value_Num:Q",
            tooltip=[
                alt.Tooltip("TimeStamp:T", title="Hora"),
                alt.Tooltip("Value_Num:Q", title="Outlier", format=",.3f"),
                alt.Tooltip("Upper_Limit:Q", title="Upper", format=",.3f"),
                alt.Tooltip("Lower_Limit:Q", title="Lower", format=",.3f"),
            ],
        )
        layers.append(outlier_pts)

    # Upper / Lower rules
    upper_rule = alt.Chart(pd.DataFrame({"y": [upper]})).mark_rule(
        color="#ef4444", strokeDash=[6, 4], strokeWidth=2,
    ).encode(y="y:Q", tooltip=[alt.Tooltip("y:Q", title="Upper Limit", format=",.3f")])

    lower_rule = alt.Chart(pd.DataFrame({"y": [lower]})).mark_rule(
        color="#ef4444", strokeDash=[6, 4], strokeWidth=2,
    ).encode(y="y:Q", tooltip=[alt.Tooltip("y:Q", title="Lower Limit", format=",.3f")])

    mean_rule = alt.Chart(pd.DataFrame({"y": [mean_val]})).mark_rule(
        color="#f59e0b", strokeDash=[8, 5], strokeWidth=2,
    ).encode(y="y:Q", tooltip=[alt.Tooltip("y:Q", title="Mean", format=",.3f")])

    layers.extend([upper_rule, lower_rule, mean_rule])

    chart = alt.layer(*layers).properties(height=350, title=f"{tag_name} — Outlier Analysis (±{std_factor}σ)").interactive()

    outlier_count = int(plot_df["Is_Outlier"].sum())
    stats = {
        "mean": mean_val, "std": std_val, "upper": upper, "lower": lower,
        "outlier_count": outlier_count, "total": len(plot_df),
        "outlier_pct": (outlier_count / len(plot_df) * 100) if len(plot_df) > 0 else 0,
    }
    return chart, stats


# ----------------------------------------------------------
# Gráfico histograma
# ----------------------------------------------------------
def _histogram_chart(df: pd.DataFrame, tag_name: str) -> alt.Chart | None:
    valid = df["Value_Num"].dropna()
    if valid.empty:
        return None
    plot_df = df.dropna(subset=["Value_Num"])
    mean_val = valid.mean()

    hist = alt.Chart(plot_df).mark_bar(color="#60a5fa", opacity=0.8).encode(
        x=alt.X("Value_Num:Q", bin=alt.Bin(maxbins=30), title="Valor"),
        y=alt.Y("count():Q", title="Frecuencia"),
    )
    mean_rule = alt.Chart(pd.DataFrame({"x": [mean_val]})).mark_rule(
        color="#f59e0b", strokeWidth=2.5
    ).encode(x="x:Q", tooltip=[alt.Tooltip("x:Q", title="Promedio", format=",.3f")])

    return alt.layer(hist, mean_rule).properties(height=280, title=f"{tag_name} — Distribución").interactive()


# ----------------------------------------------------------
# Determinar si hay instrucciones de un tipo
# ----------------------------------------------------------
def _has_instruction(chart_instructions: list, instruction_type: str) -> bool:
    for instr in chart_instructions:
        if isinstance(instr, dict) and instr.get("type", "").lower() == instruction_type.lower():
            return True
    return False


# ----------------------------------------------------------
# Dashboard completo para una máquina
# ----------------------------------------------------------
def render_machine_dashboard(
    machine_name: str,
    tag_data: dict[str, pd.DataFrame],
    time_range_label: str = "hoy",
    chart_instructions: list | None = None,
):
    """
    tag_data: dict {tag_name: DataFrame con columnas estándar}
    """
    if chart_instructions is None:
        chart_instructions = []

    show_outliers = _has_instruction(chart_instructions, "outliers")
    show_histogram = _has_instruction(chart_instructions, "histogram")

    if not tag_data:
        st.warning(f"No se obtuvieron datos para **{machine_name}**.")
        return

    # Header
    st.markdown(f"### 📊 {machine_name}")
    st.caption(f"Período: {time_range_label}")

    # KPIs globales
    total_points = sum(len(df) for df in tag_data.values())
    numeric_tags = 0
    categorical_tags = 0

    for df in tag_data.values():
        mode = infer_tag_mode(df)
        if mode == "numeric":
            numeric_tags += 1
        else:
            categorical_tags += 1

    cols = st.columns(4)
    cols[0].metric("Tags consultados", len(tag_data))
    cols[1].metric("Puntos totales", f"{total_points:,}")
    cols[2].metric("Tags numéricos", numeric_tags)
    cols[3].metric("Tags categóricos", categorical_tags)

    # Separar numéricos y categóricos
    numeric_items = []
    categorical_items = []

    for tag_name, df in tag_data.items():
        if df.empty:
            continue
        mode = infer_tag_mode(df)
        if mode == "numeric":
            numeric_items.append((tag_name, df))
        else:
            categorical_items.append((tag_name, df))

    # --- Sección numérica ---
    if numeric_items:
        st.markdown("#### 📈 Parámetros numéricos")

        for tag_name, df in numeric_items:
            with st.container(border=True):
                valid = df["Value_Num"].dropna()

                mc = st.columns(5)
                mc[0].metric("Tag", tag_name.split(".")[-1] if "." in tag_name else tag_name)
                mc[1].metric("Último", _safe_metric(valid.iloc[-1] if not valid.empty else None))
                mc[2].metric("Mín", _safe_metric(valid.min() if not valid.empty else None))
                mc[3].metric("Máx", _safe_metric(valid.max() if not valid.empty else None))
                mc[4].metric("Promedio", _safe_metric(valid.mean() if not valid.empty else None))

                if show_outliers:
                    # Gráfico con límites de desviación estándar
                    outlier_chart, stats = _numeric_outlier_chart(df, tag_name)
                    if outlier_chart is not None:
                        chart_with_ai(outlier_chart, df=df, chart_id=f"outlier_{tag_name}",
                                      context={"tag": tag_name, "type": "outlier_chart"})
                        # Métricas de outliers
                        oc = st.columns(4)
                        oc[0].metric("σ (Std Dev)", _safe_metric(stats.get("std")))
                        oc[1].metric("Upper Limit", _safe_metric(stats.get("upper")))
                        oc[2].metric("Lower Limit", _safe_metric(stats.get("lower")))
                        oc[3].metric("Outliers", f"{stats.get('outlier_count', 0)} ({stats.get('outlier_pct', 0):.1f}%)")
                    else:
                        st.info("Sin datos suficientes para análisis de outliers.")
                else:
                    chart = _numeric_chart(df.dropna(subset=["TimeStamp", "Value_Num"]), tag_name)
                    chart_with_ai(chart, df=df, chart_id=f"numeric_{tag_name}",
                                  context={"tag": tag_name, "type": "numeric_timeseries"})

                if show_histogram:
                    hist = _histogram_chart(df, tag_name)
                    if hist is not None:
                        chart_with_ai(hist, df=df, chart_id=f"hist_{tag_name}",
                                      context={"tag": tag_name, "type": "histogram"})

    # --- Sección categórica ---
    if categorical_items:
        st.markdown("#### States & Signals")

        for tag_name, df in categorical_items:
            mode = infer_tag_mode(df)
            with st.container(border=True):
                short_name = tag_name.split(".")[-1] if "." in tag_name else tag_name

                # Determinar la mejor columna de valores disponible
                _NULLS = {"nan", "none", "null", "", "na", "n/a"}
                def _best_val(row):
                    for col in ("Value_Str", "Value_Raw"):
                        v = str(row.get(col, "")).strip()
                        if v.lower() not in _NULLS:
                            return v
                    if pd.notna(row.get("Value_Num")):
                        n = row["Value_Num"]
                        return str(int(n)) if float(n) == int(float(n)) else f"{n:.4g}"
                    return None

                # Valor actual: ultima fila con datos
                current = "N/A"
                total_pts = len(df)
                for _, row in df.sort_values("TimeStamp", ascending=False).iterrows():
                    v = _best_val(row)
                    if v:
                        current = v
                        break

                # Estados únicos (de cualquier columna de valor)
                val_series = None
                for col in ("Value_Str", "Value_Raw"):
                    if col in df.columns:
                        s = df[col].astype(str).str.strip()
                        s = s[~s.str.lower().isin(_NULLS)]
                        if s.notna().sum() > 0:
                            val_series = s
                            break
                if val_series is None and "Value_Num" in df.columns:
                    val_series = df["Value_Num"].dropna().apply(
                        lambda n: str(int(n)) if float(n) == int(float(n)) else f"{n:.4g}"
                    )

                unique_count = val_series.nunique() if val_series is not None else 0

                mc = st.columns(4)
                mc[0].metric("Tag", short_name)
                mc[1].metric("Valor actual", str(current))
                mc[2].metric("Estados distintos", unique_count)
                mc[3].metric("Puntos", f"{total_pts:,}")

                # % de tiempo por estado si pocos estados
                if val_series is not None and 0 < unique_count <= 8:
                    state_pct = (val_series.value_counts(normalize=True) * 100).round(1).to_dict()
                    pct_cols = st.columns(min(unique_count, 4))
                    for i, (state, pct) in enumerate(list(state_pct.items())[:4]):
                        pct_cols[i % len(pct_cols)].metric(f"• {state}", f"{pct}%")

                # Pasar el DF completo — _smart_categorical_chart resuelve la columna internamente
                chart = _smart_categorical_chart(df, tag_name, mode)
                if chart is not None:
                    chart_with_ai(chart, df=df, chart_id=f"cat_{tag_name}",
                                  context={"tag": tag_name, "type": "categorical", "mode": mode})
                elif val_series is not None and not val_series.empty:
                    # Fallback: tabla de frecuencias
                    freq = val_series.value_counts().reset_index()
                    freq.columns = ["Estado", "Ocurrencias"]
                    st.dataframe(freq, use_container_width=True, hide_index=True)
                else:
                    st.caption("Sin valores registrados en este periodo.")


# ----------------------------------------------------------
# Dashboard de un solo atributo (drill-down)
# ----------------------------------------------------------
def render_attribute_detail(
    tag_name: str,
    df: pd.DataFrame,
):
    if df.empty:
        st.warning(f"Sin datos para **{tag_name}**")
        return

    mode = infer_tag_mode(df)

    st.markdown(f"### 🔍 Detalle: {tag_name}")

    if mode == "numeric":
        valid = df["Value_Num"].dropna()
        timestamps = df["TimeStamp"].dropna()

        cols = st.columns(6)
        cols[0].metric("Puntos", f"{len(valid):,}")
        cols[1].metric("Último", _safe_metric(valid.iloc[-1] if not valid.empty else None))
        cols[2].metric("Mín", _safe_metric(valid.min() if not valid.empty else None))
        cols[3].metric("Máx", _safe_metric(valid.max() if not valid.empty else None))
        cols[4].metric("Promedio", _safe_metric(valid.mean() if not valid.empty else None))
        cols[5].metric("Desv. Std", _safe_metric(valid.std() if not valid.empty else None))

        chart = _numeric_chart(df.dropna(subset=["TimeStamp", "Value_Num"]), tag_name)
        chart_with_ai(chart, df=df, chart_id=f"detail_num_{tag_name}",
                      context={"tag": tag_name, "type": "numeric_detail"})

        # Histograma
        if not valid.empty:
            hist = (
                alt.Chart(df.dropna(subset=["Value_Num"]))
                .mark_bar(color="#60a5fa", opacity=0.8)
                .encode(
                    x=alt.X("Value_Num:Q", bin=alt.Bin(maxbins=30), title="Valor"),
                    y=alt.Y("count():Q", title="Frecuencia"),
                )
                .properties(height=250, title="Distribución")
            )
            chart_with_ai(hist, df=df, chart_id=f"detail_hist_{tag_name}",
                          context={"tag": tag_name, "type": "histogram"})

    else:
        current = df["Value_Str"].dropna().iloc[-1] if df["Value_Str"].notna().any() else "N/A"
        cols = st.columns(3)
        cols[0].metric("Puntos", f"{len(df):,}")
        cols[1].metric("Valor actual", str(current))
        cols[2].metric("Estados únicos", df["Value_Str"].nunique(dropna=True))

        chart = _categorical_chart(df.dropna(subset=["TimeStamp", "Value_Str"]), tag_name)
        chart_with_ai(chart, df=df, chart_id=f"detail_cat_{tag_name}",
                      context={"tag": tag_name, "type": "categorical_detail"})

    # Tabla de datos
    with st.expander("📋 Ver datos crudos"):
        show_cols = [c for c in ["TimeStamp", "Value_Raw", "Value_Num", "Value_Str", "Tag_Type"] if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True, height=350)


# ----------------------------------------------------------
# Lista de máquinas como botones seleccionables
# ----------------------------------------------------------
def render_machine_list(machines_df: pd.DataFrame) -> str | None:
    """
    Muestra las máquinas agrupadas por línea como botones.
    Devuelve el nombre de la máquina si el usuario hizo clic en una, o None.
    """
    if machines_df.empty:
        st.info("No se encontraron máquinas.")
        return None

    selected = None
    for line_name, group in machines_df.groupby("line_name"):
        st.caption(line_name)
        btn_cols = st.columns(min(4, len(group)))
        for i, (_, row) in enumerate(group.iterrows()):
            with btn_cols[i % len(btn_cols)]:
                if st.button(
                    row["machine_name"],
                    key=f"ml_{line_name}_{row['machine_name']}",
                    use_container_width=True,
                ):
                    selected = row["machine_name"]
    return selected


def build_data_summary_text(tag_data: dict[str, pd.DataFrame]) -> str:
    """
    Genera un resumen de texto de los datos para pasarle al LLM.
    """
    parts = []
    for tag_name, df in tag_data.items():
        if df.empty:
            parts.append(f"- {tag_name}: sin datos")
            continue

        mode = infer_tag_mode(df)
        if mode == "numeric":
            valid = df["Value_Num"].dropna()
            if valid.empty:
                parts.append(f"- {tag_name}: sin valores numéricos válidos")
            else:
                parts.append(
                    f"- {tag_name}: último={valid.iloc[-1]:.3f}, "
                    f"min={valid.min():.3f}, max={valid.max():.3f}, "
                    f"promedio={valid.mean():.3f}, puntos={len(valid)}"
                )
        else:
            current = df["Value_Str"].dropna().iloc[-1] if df["Value_Str"].notna().any() else "N/A"
            parts.append(f"- {tag_name}: valor actual='{current}', puntos={len(df)}")

    return "\n".join(parts) if parts else "Sin datos disponibles."
