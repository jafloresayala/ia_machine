"""
Chart AI — envoltura reutilizable que agrega "modo IA" a cualquier grafica Altair.

Permite al usuario:
  1. Modificar visualmente la grafica (agregar lineas, bandas, anotaciones) via LLM
  2. Hacer preguntas en lenguaje natural sobre los datos de la grafica

Uso tipico:
    from chart_ai import chart_with_ai
    chart_with_ai(chart, df=df, chart_id="mi_grafica", context={"tag": "Temperature"})
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from llm_engine import chat_completion_json, chat_completion


# ==================================================================
# Sistema prompts
# ==================================================================
_OVERLAY_SYSTEM = """Eres un analista de datos industriales y cientifico de datos experto.
Puedes modificar graficas de series de tiempo de formas ARBITRARIAS: estadistica, modelos predictivos, suavizado, control de calidad, regresiones, etc.
Responde SIEMPRE con JSON valido. Sin texto fuera del JSON.

Esquema de respuesta:
{
  "overlays": [ <lista de capas, ver tipos abajo> ],
  "explanation": "descripcion breve de lo que calculaste"
}

== TIPOS DE OVERLAY DISPONIBLES ==

1. LINEA HORIZONTAL fija:
   {"type": "hline", "value": 123.45, "label": "Media", "color": "#10b981", "dashed": false}

2. BANDA HORIZONTAL:
   {"type": "band", "lower": 90.0, "upper": 110.0, "label": "Rango control", "color": "#f59e0b", "opacity": 0.15}

3. LINEA VERTICAL en tiempo:
   {"type": "vline", "timestamp": "2026-04-20T08:30:00", "label": "Evento", "color": "#ef4444", "dashed": true}

4. SERIE CALCULADA (el mas poderoso — USALO para predicciones, modelos, suavizado, etc.):
   {
     "type": "series",
     "data": [{"t": "2026-04-20T08:00:00", "y": 123.4}, {"t": "2026-04-20T09:00:00", "y": 124.1}, ...],
     "label": "Prediccion 5 dias",
     "color": "#8b5cf6",
     "dashed": true,
     "width": 2
   }
   IMPORTANTE: Calcula TU MISMO los valores de 'data'. Usa regresion lineal, exponencial, media movil, o cualquier modelo que se ajuste mejor a los datos provistos en 'recent_data'.
   Para predicciones futuras: extrapola desde el ultimo timestamp en 'recent_data' usando el intervalo en 'sample_interval_seconds'.

5. TENDENCIA LINEAL (Altair calcula la regresion automaticamente):
   {"type": "trendline", "color": "#3b82f6", "label": "Tendencia"}

6. MEDIA MOVIL (calculada en Python automaticamente):
   {"type": "moving_avg", "window": 10, "color": "#f59e0b", "label": "Media movil"}

7. SUAVIZADO EXPONENCIAL (calculado en Python):
   {"type": "ewm", "span": 5, "color": "#06b6d4", "label": "Suavizado exp."}

== COMO HACER PREDICCIONES ==
Cuando el usuario pida predecir, pronosticar o modelar valores futuros:
1. Analiza los ultimos valores en 'recent_data' para detectar tendencia (subida, bajada, estable, ciclica).
2. Calcula la pendiente de los ultimos puntos con regresion lineal simple: pendiente = (y_ultimo - y_primero) / n_puntos.
3. Genera los puntos futuros: cada punto = ultimo_y + pendiente * i, con timestamp = ultimo_t + sample_interval_seconds * i.
4. Para 5 dias: calcula cuantos intervalos caben en 5 dias = (5*86400) / sample_interval_seconds.
5. Genera la serie con esos puntos y ponla en 'data' del overlay tipo 'series'.
6. Puedes incluir tambien bandas de incertidumbre con tipo 'band' usando +/- 1 sigma como margen.

== REGLAS ==
- Usa SIEMPRE los valores numericos exactos de 'numeric_stats' y 'recent_data' del usuario.
- NUNCA pongas formulas como "mean + 2*std" — haz la aritmetica tu mismo.
- Puedes combinar multiples overlays en el array.
- Si la peticion no es clara, haz lo mas util posible y explica en 'explanation'.
"""

_QA_SYSTEM = """Eres un analista de datos industriales SMT experto en estadistica y control de procesos.
Responde preguntas sobre la grafica y sus datos usando SIEMPRE los valores exactos del resumen provisto.
Si los datos incluyen una matriz de correlacion, interpreta los valores de Pearson (rho): >0.7 correlacion fuerte positiva, 0.4-0.7 moderada, <0.2 debil o nula, negativo = correlacion inversa.
Si los datos incluyen un histograma, comenta la distribucion, simetria, outliers y normalidad.
Si los datos son series de tiempo, comenta tendencia, estabilidad, variabilidad y anomalias.
Se conciso y tecnico. Usa los valores reales del resumen. Responde en espanol."""

_FORECAST_SYSTEM = """Eres un experto en series de tiempo industriales (SPC, manufactura SMT, procesos continuos).
Analizas datos historicos y eliges el modelo predictivo mas apropiado.
Responde SOLO con JSON valido, sin texto adicional.

Esquema de respuesta:
{
  "model": "linear" | "polynomial" | "holt" | "seasonal",
  "params": {
    "degree": 2,
    "alpha": 0.3,
    "beta": 0.1,
    "periods_back": 3
  },
  "reasoning": "descripcion breve del patron detectado y por que elegiste este modelo",
  "trend_direction": "subiendo" | "bajando" | "estable" | "ciclico" | "variable"
}

GUIA de seleccion:
- linear: tendencia monotona clara (sube o baja consistentemente sin curvatura)
- polynomial: curvatura notable (aceleracion, desaceleracion, cambio de tendencia en U o invertida)
- holt: tendencia con suavizado (ruido industrial, tendencia suave que cambia gradualmente) -- OPCION PREFERIDA para series industriales
- seasonal: patron ciclico o repetitivo visible en las muestras (turnos de trabajo, ciclos de produccion, temperatura del dia)

PARAMETROS:
- polynomial.degree: 2 para parabolica, 3 para cubica (raramente 4)
- holt.alpha: 0.1 (muy suave) a 0.9 (sigue cada cambio). Defecto: 0.3
- holt.beta: 0.05 (tendencia estable) a 0.3 (tendencia cambiante). Defecto: 0.1
- seasonal.periods_back: cuantos ciclos hacia atras usar para proyectar (1-5)

Analiza 'sample_values' para detectar si los datos suben/bajan/oscilan/tienen curvatura.
"""


# ==================================================================
# Construccion de overlays
# ==================================================================
def _parse_horizon_seconds(prompt: str) -> float | None:
    """Extrae el horizonte temporal de un prompt.
    Soporta: dias, horas, minutos, semanas. Retorna segundos, o None si no detecta.
    """
    import re
    p = prompt.lower()
    # Buscar patrones como "5 dias", "2 horas", "30 minutos", "1 semana"
    patterns = [
        (r"(\d+(?:[.,]\d+)?)\s*(?:dia|día|day)s?", 86400),
        (r"(\d+(?:[.,]\d+)?)\s*(?:hora|hour|hr)s?", 3600),
        (r"(\d+(?:[.,]\d+)?)\s*(?:minuto|minute|min)s?", 60),
        (r"(\d+(?:[.,]\d+)?)\s*(?:semana|week|wk)s?", 7 * 86400),
        (r"(\d+(?:[.,]\d+)?)\s*(?:mes|month)(?:es)?", 30 * 86400),
    ]
    for regex, unit_sec in patterns:
        m = re.search(regex, p)
        if m:
            try:
                return float(m.group(1).replace(",", ".")) * unit_sec
            except ValueError:
                continue
    return None


def _build_forecast_overlay(prompt: str, df: pd.DataFrame | None) -> list[dict] | None:
    """Calcula una prediccion lineal (regresion) server-side sobre df.
    Retorna overlays de tipo 'series' (prediccion) + 'band' (incertidumbre).
    """
    if df is None or df.empty:
        return None
    if "Value_Num" not in df.columns or "TimeStamp" not in df.columns:
        return None

    d = df.dropna(subset=["Value_Num", "TimeStamp"]).copy()
    if len(d) < 5:
        return None
    d["TimeStamp"] = pd.to_datetime(d["TimeStamp"], errors="coerce")
    d = d.dropna(subset=["TimeStamp"]).sort_values("TimeStamp")
    if len(d) < 5:
        return None

    # Horizonte en segundos (por defecto: igual al rango historico, o 1 dia si es mas)
    horizon_s = _parse_horizon_seconds(prompt)
    t0 = d["TimeStamp"].iloc[0]
    t_last = d["TimeStamp"].iloc[-1]
    hist_span_s = (t_last - t0).total_seconds()
    if horizon_s is None:
        horizon_s = max(hist_span_s, 86400)

    # Regresion lineal: y = a*x + b  (x = segundos desde t0)
    secs = (d["TimeStamp"] - t0).dt.total_seconds().values
    yval = d["Value_Num"].values
    try:
        slope, intercept = np.polyfit(secs, yval, 1)
    except Exception:
        return None

    # Residuos para incertidumbre
    predictions_hist = slope * secs + intercept
    residuals = yval - predictions_hist
    sigma = float(np.std(residuals, ddof=0))

    # Generar 60 puntos futuros distribuidos uniformemente en el horizonte
    n_pts = 60
    step_s = horizon_s / n_pts
    last_sec = secs[-1]
    future_points = []
    future_lower = []
    future_upper = []
    for i in range(1, n_pts + 1):
        x = last_sec + step_s * i
        y = slope * x + intercept
        t = t0 + pd.Timedelta(seconds=x)
        # Intervalo de confianza se expande con la distancia desde los datos
        widening = 1.0 + (step_s * i) / max(hist_span_s, 1.0) * 0.5
        ci = 1.96 * sigma * widening
        future_points.append({"t": t.isoformat(), "y": round(float(y), 4)})
        future_lower.append({"t": t.isoformat(), "y": round(float(y - ci), 4)})
        future_upper.append({"t": t.isoformat(), "y": round(float(y + ci), 4)})

    # Incluir el ultimo punto real para continuidad visual
    anchor = {"t": t_last.isoformat(), "y": round(float(yval[-1]), 4)}
    future_points.insert(0, anchor)

    # Etiqueta del horizonte en formato legible
    if horizon_s >= 86400 * 7:
        horizon_label = f"{horizon_s / (86400 * 7):g} sem"
    elif horizon_s >= 86400:
        horizon_label = f"{horizon_s / 86400:g} d"
    elif horizon_s >= 3600:
        horizon_label = f"{horizon_s / 3600:g} h"
    else:
        horizon_label = f"{horizon_s / 60:g} min"

    trend_dir = "sube" if slope > 0 else "baja" if slope < 0 else "estable"
    return [
        # Banda de incertidumbre (IC 95%)
        {
            "type": "series_band",
            "lower": future_lower,
            "upper": future_upper,
            "color": "#8b5cf6",
            "opacity": 0.15,
            "label": f"IC 95%",
        },
        # Linea predictiva
        {
            "type": "series",
            "data": future_points,
            "color": "#8b5cf6",
            "dashed": True,
            "width": 2.5,
            "label": f"Prediccion {horizon_label} ({trend_dir})",
        },
        # Separador vertical en el ultimo dato real
        {
            "type": "vline",
            "timestamp": t_last.isoformat(),
            "color": "#8b5cf6",
            "dashed": True,
            "label": "Inicio prediccion",
        },
    ]


def _local_overlay_fallback(prompt: str, summary: dict, df: pd.DataFrame | None = None) -> list[dict] | None:
    """
    Intenta generar overlays desde el prompt sin LLM, reconociendo patrones comunes.
    Retorna lista de overlays, o None si no reconocio el patron.
    """
    import re
    p = prompt.lower().strip()
    stats = summary.get("numeric_stats") or {}
    mean = stats.get("mean")
    std = stats.get("std")
    vmin = stats.get("min")
    vmax = stats.get("max")
    median = stats.get("median")

    overlays: list[dict] = []

    # Detectar numero de sigmas en el prompt: "sigma 1.5", "2 sigma", "+/- 3 desviaciones"
    sigma_match = re.search(
        r"(?:sigma|desviaci[oó]n(?:es)?|std)\s*(?:de\s+)?(\d+(?:[.,]\d+)?)"
        r"|(\d+(?:[.,]\d+)?)\s*(?:sigma|desviaci[oó]n(?:es)?|std)"
        r"|\+\-?\s*(\d+(?:[.,]\d+)?)",
        p,
    )
    sigmas = None
    if sigma_match:
        grp = next((g for g in sigma_match.groups() if g), None)
        if grp:
            try:
                sigmas = float(grp.replace(",", "."))
            except ValueError:
                sigmas = None

    wants_band = any(w in p for w in ("banda", "band", "franja", "rango", "zona"))
    wants_limits = any(w in p for w in ("limite", "limits", "upper", "lower", "superior", "inferior", "control"))
    wants_mean = any(w in p for w in ("promedio", "mean", "media", "average"))
    wants_median = "median" in p
    wants_max = any(w in p for w in ("maximo", "máximo", "max ", "maximum"))
    wants_min = any(w in p for w in ("minimo", "mínimo", "min ", "minimum"))
    wants_trend = any(w in p for w in ("tendencia", "trend", "regresion", "regresión"))
    wants_range = "rango " in p and vmin is not None and vmax is not None

    # Sigma (banda o limites)
    if sigmas is not None and mean is not None and std is not None:
        lo = mean - sigmas * std
        hi = mean + sigmas * std
        if wants_band or (not wants_limits):
            overlays.append({
                "type": "band", "lower": lo, "upper": hi,
                "label": f"Control +/-{sigmas:g}σ", "color": "#f59e0b", "opacity": 0.15,
            })
        else:
            overlays.append({"type": "hline", "value": hi, "label": f"Upper +{sigmas:g}σ", "color": "#ef4444", "dashed": True})
            overlays.append({"type": "hline", "value": lo, "label": f"Lower -{sigmas:g}σ", "color": "#ef4444", "dashed": True})
            overlays.append({"type": "hline", "value": mean, "label": "Mean", "color": "#10b981", "dashed": False})

    if wants_mean and mean is not None and not any(o.get("label") == "Mean" for o in overlays):
        overlays.append({"type": "hline", "value": mean, "label": "Mean", "color": "#10b981", "dashed": False})
    if wants_median and median is not None:
        overlays.append({"type": "hline", "value": median, "label": "Mediana", "color": "#8b5cf6", "dashed": True})
    if wants_max and vmax is not None:
        overlays.append({"type": "hline", "value": vmax, "label": "Max", "color": "#ef4444", "dashed": False})
    if wants_min and vmin is not None:
        overlays.append({"type": "hline", "value": vmin, "label": "Min", "color": "#3b82f6", "dashed": False})
    if wants_trend:
        overlays.append({"type": "trendline", "color": "#3b82f6", "label": "Tendencia"})

    # ---- Modelos predictivos ----
    wants_forecast = any(w in p for w in ("predic", "forecast", "pronostic", "proyecc", "futuro", "proximo", "next "))
    wants_ewm = any(w in p for w in ("exponencial", "ewm", "ema", "suavizado"))
    wants_ma  = any(w in p for w in ("media movil", "moving avg", "rolling", "promedio movil", "suaviza", "smooth"))

    num_match = re.search(r"(\d+)\s*(?:puntos?|pts?|ventana|window|periodos?|muestras?)", p)
    num_val   = int(num_match.group(1)) if num_match else None

    # ---- Prediccion con regresion lineal (server-side) ----
    if wants_forecast:
        pred_overlay = _build_forecast_overlay(prompt, df)
        if pred_overlay is not None:
            overlays.extend(pred_overlay)
    if wants_ewm:
        span = num_val or 5
        overlays.append({"type": "ewm", "span": span, "color": "#06b6d4", "label": f"Suavizado exp.(span={span})"})
    elif wants_ma:
        window = num_val or 10
        overlays.append({"type": "moving_avg", "window": window, "color": "#f59e0b", "label": f"Media movil ({window}pts)"})

    return overlays if overlays else None


def _build_summary(df: pd.DataFrame | None, context: dict | None) -> dict:
    """Resumen compacto de estadisticas para enviar al LLM."""
    summary = dict(context or {})
    if df is None or df.empty:
        return summary

    cols = set(df.columns)

    # ---- Matriz de correlacion (param_a, param_b, corr) ----
    if {"param_a", "param_b", "corr"}.issubset(cols):
        valid = df.dropna(subset=["corr"])
        # Pares unicos (sin duplicados ni diagonal)
        pairs = (
            valid[valid["param_a"] != valid["param_b"]]
            .copy()
            .assign(key=lambda d: d.apply(
                lambda r: tuple(sorted([r["param_a"], r["param_b"]])), axis=1
            ))
            .drop_duplicates(subset="key")
            .drop(columns="key")
            .sort_values("corr", key=abs, ascending=False)
        )
        summary["correlation_pairs"] = [
            {"A": row["param_a"], "B": row["param_b"], "rho": round(float(row["corr"]), 4)}
            for _, row in pairs.iterrows()
        ]
        top_pos = pairs[pairs["corr"] > 0].head(5)
        top_neg = pairs[pairs["corr"] < 0].head(5)
        if not top_pos.empty:
            summary["top_positive_correlations"] = [
                {"A": r["param_a"], "B": r["param_b"], "rho": round(float(r["corr"]), 4)}
                for _, r in top_pos.iterrows()
            ]
        if not top_neg.empty:
            summary["top_negative_correlations"] = [
                {"A": r["param_a"], "B": r["param_b"], "rho": round(float(r["corr"]), 4)}
                for _, r in top_neg.iterrows()
            ]
        params = sorted(set(valid["param_a"].tolist() + valid["param_b"].tolist()))
        summary["parameters"] = params
        summary["n_params"] = len(params)
        return summary

    # ---- Serie de tiempo (TimeStamp + Value_Num) ----
    if "TimeStamp" in cols:
        ts = pd.to_datetime(df["TimeStamp"], errors="coerce").dropna()
        if not ts.empty:
            summary["time_start"] = ts.min().isoformat()
            summary["time_end"] = ts.max().isoformat()
            summary["n_points"] = int(len(ts))

    if "Value_Num" in cols:
        v = df["Value_Num"].dropna()
        if not v.empty:
            summary["numeric_stats"] = {
                "mean": round(float(v.mean()), 4),
                "std":  round(float(v.std(ddof=0)), 4),
                "min":  round(float(v.min()), 4),
                "max":  round(float(v.max()), 4),
                "median": round(float(v.median()), 4),
                "last": round(float(v.iloc[-1]), 4),
                "count": int(len(v)),
                "cv_pct": round(float(v.std(ddof=0) / v.mean() * 100), 2) if v.mean() != 0 else None,
            }
            # Muestra distribucion para histogramas
            if context and context.get("type") == "histogram":
                counts, edges = pd.cut(v, bins=min(20, max(5, len(v)//10)), retbins=True)
                freq = counts.value_counts(sort=False)
                summary["histogram_bins"] = [
                    {"range": f"{edges[i]:.3g}-{edges[i+1]:.3g}", "count": int(c)}
                    for i, c in enumerate(freq)
                ]

        # recent_data: ultimos 80 puntos (o submuestreados a 80) para que el LLM pueda calcular modelos
        if "TimeStamp" in cols:
            d = df.dropna(subset=["Value_Num", "TimeStamp"]).copy()
            d["TimeStamp"] = pd.to_datetime(d["TimeStamp"], errors="coerce")
            d = d.sort_values("TimeStamp")
            if len(d) > 80:
                # Submuestrear uniformemente a 80 puntos
                idx = np.linspace(0, len(d) - 1, 80, dtype=int)
                d = d.iloc[idx]
            if not d.empty:
                summary["recent_data"] = [
                    {"t": row["TimeStamp"].isoformat(), "y": round(float(row["Value_Num"]), 4)}
                    for _, row in d.iterrows()
                ]
                # Intervalo tipico entre muestras en segundos
                if len(d) >= 2:
                    intervals = d["TimeStamp"].diff().dt.total_seconds().dropna()
                    summary["sample_interval_seconds"] = round(float(intervals.median()), 1)

    # ---- Categoricos ----
    if "Value_Str" in cols and "Value_Num" not in cols:
        v = df["Value_Str"].dropna().astype(str)
        if not v.empty:
            top = v.value_counts().head(5).to_dict()
            summary["categorical_top"] = {str(k): int(cnt) for k, cnt in top.items()}
            summary["unique_states"] = int(v.nunique())

    return summary


def _apply_overlays(base_chart: alt.Chart, overlays: list[dict], df: pd.DataFrame | None) -> alt.Chart:
    """Aplica la lista de overlays sobre la grafica base, retornando un LayerChart."""
    if not overlays:
        return base_chart

    layers = [base_chart]

    for ov in overlays:
        try:
            ov_type = str(ov.get("type", "")).lower()
            color = ov.get("color") or "#ef4444"
            label = str(ov.get("label") or "")
            dashed = bool(ov.get("dashed", False))
            stroke_dash = [5, 5] if dashed else [1, 0]

            if ov_type == "hline" and "value" in ov:
                y_val = float(ov["value"])
                rule_df = pd.DataFrame({"y": [y_val]})
                rule = alt.Chart(rule_df).mark_rule(color=color, strokeDash=stroke_dash, strokeWidth=2).encode(y="y:Q")
                layers.append(rule)
                if label:
                    txt = alt.Chart(rule_df).mark_text(
                        align="left", dx=6, dy=-6, color=color, fontSize=11, fontWeight="bold"
                    ).encode(y="y:Q", text=alt.value(f"{label}: {y_val:g}"))
                    layers.append(txt)

            elif ov_type == "band" and "lower" in ov and "upper" in ov:
                lo, hi = float(ov["lower"]), float(ov["upper"])
                opacity = float(ov.get("opacity", 0.12))
                band_df = pd.DataFrame({"lower": [lo], "upper": [hi]})
                band = alt.Chart(band_df).mark_rect(color=color, opacity=opacity).encode(
                    y="lower:Q", y2="upper:Q"
                )
                layers.append(band)
                if label:
                    txt = alt.Chart(pd.DataFrame({"y": [(lo + hi) / 2]})).mark_text(
                        align="right", dx=-6, color=color, fontSize=11, fontWeight="bold"
                    ).encode(y="y:Q", text=alt.value(label))
                    layers.append(txt)

            elif ov_type == "vline" and "timestamp" in ov:
                ts = pd.to_datetime(ov["timestamp"], errors="coerce")
                if pd.isna(ts):
                    continue
                vrule_df = pd.DataFrame({"x": [ts]})
                vrule = alt.Chart(vrule_df).mark_rule(color=color, strokeDash=stroke_dash, strokeWidth=2).encode(x="x:T")
                layers.append(vrule)
                if label:
                    txt = alt.Chart(vrule_df).mark_text(
                        align="left", dx=6, dy=-100, color=color, fontSize=11, fontWeight="bold", angle=270
                    ).encode(x="x:T", text=alt.value(label))
                    layers.append(txt)

            elif ov_type == "series" and "data" in ov:
                # Serie calculada por el LLM: [{"t": ISO_str, "y": float}, ...]
                raw_data = ov["data"]
                if isinstance(raw_data, list) and len(raw_data) >= 2:
                    series_df = pd.DataFrame(raw_data)
                    series_df["t"] = pd.to_datetime(series_df["t"], errors="coerce")
                    series_df["y"] = pd.to_numeric(series_df["y"], errors="coerce")
                    series_df = series_df.dropna()
                    if not series_df.empty:
                        stroke_w = float(ov.get("width", 2))
                        s_line = alt.Chart(series_df).mark_line(
                            color=color, strokeDash=stroke_dash, strokeWidth=stroke_w, opacity=0.9
                        ).encode(x="t:T", y="y:Q")
                        layers.append(s_line)
                        if label:
                            last = series_df.iloc[-1]
                            txt_df = pd.DataFrame({"x": [last["t"]], "y": [last["y"]]})
                            txt = alt.Chart(txt_df).mark_text(
                                align="left", dx=6, dy=-6, color=color,
                                fontSize=11, fontWeight="bold"
                            ).encode(x="x:T", y="y:Q", text=alt.value(label))
                            layers.append(txt)

            elif ov_type == "series_band" and "lower" in ov and "upper" in ov:
                # Banda de incertidumbre: lower y upper son listas [{"t", "y"}, ...]
                lo_data = ov["lower"]
                hi_data = ov["upper"]
                if (isinstance(lo_data, list) and isinstance(hi_data, list)
                        and len(lo_data) == len(hi_data) and len(lo_data) >= 2):
                    band_df = pd.DataFrame({
                        "t": [pd.to_datetime(p["t"], errors="coerce") for p in lo_data],
                        "lower": [float(p["y"]) for p in lo_data],
                        "upper": [float(p["y"]) for p in hi_data],
                    }).dropna()
                    if not band_df.empty:
                        opacity = float(ov.get("opacity", 0.15))
                        area = alt.Chart(band_df).mark_area(
                            color=color, opacity=opacity
                        ).encode(x="t:T", y="lower:Q", y2="upper:Q")
                        layers.append(area)

            elif ov_type == "trendline" and df is not None and "Value_Num" in df.columns and "TimeStamp" in df.columns:
                d = df.dropna(subset=["Value_Num", "TimeStamp"]).copy()
                if len(d) >= 2:
                    d["TimeStamp"] = pd.to_datetime(d["TimeStamp"], errors="coerce")
                    trend = alt.Chart(d).transform_regression(
                        "TimeStamp", "Value_Num"
                    ).mark_line(color=color, strokeDash=[6, 3], strokeWidth=2).encode(
                        x="TimeStamp:T", y="Value_Num:Q"
                    )
                    layers.append(trend)

            elif ov_type == "moving_avg" and df is not None and "Value_Num" in df.columns and "TimeStamp" in df.columns:
                d = df.dropna(subset=["Value_Num", "TimeStamp"]).copy()
                d["TimeStamp"] = pd.to_datetime(d["TimeStamp"], errors="coerce")
                d = d.sort_values("TimeStamp")
                window = max(2, int(ov.get("window", 10)))
                if len(d) >= window:
                    d["_ma"] = d["Value_Num"].rolling(window=window, min_periods=2, center=True).mean()
                    plot_d = d.dropna(subset=["_ma"])[["TimeStamp", "_ma"]].rename(columns={"_ma": "Value_Num"})
                    if not plot_d.empty:
                        ma_line = alt.Chart(plot_d).mark_line(
                            color=color, strokeWidth=2, opacity=0.9
                        ).encode(x="TimeStamp:T", y="Value_Num:Q")
                        layers.append(ma_line)
                        if label:
                            last = plot_d.iloc[-1]
                            txt_df = pd.DataFrame({"x": [last["TimeStamp"]], "y": [last["Value_Num"]]})
                            txt = alt.Chart(txt_df).mark_text(
                                align="left", dx=6, dy=-6, color=color, fontSize=11, fontWeight="bold"
                            ).encode(x="x:T", y="y:Q", text=alt.value(label))
                            layers.append(txt)

            elif ov_type == "ewm" and df is not None and "Value_Num" in df.columns and "TimeStamp" in df.columns:
                d = df.dropna(subset=["Value_Num", "TimeStamp"]).copy()
                d["TimeStamp"] = pd.to_datetime(d["TimeStamp"], errors="coerce")
                d = d.sort_values("TimeStamp")
                span = max(2, int(ov.get("span", 5)))
                if len(d) >= 3:
                    d["_ewm"] = d["Value_Num"].ewm(span=span, adjust=False).mean()
                    plot_d = d[["TimeStamp", "_ewm"]].rename(columns={"_ewm": "Value_Num"})
                    ewm_line = alt.Chart(plot_d).mark_line(
                        color=color, strokeWidth=2, opacity=0.9
                    ).encode(x="TimeStamp:T", y="Value_Num:Q")
                    layers.append(ewm_line)
                    if label:
                        last = plot_d.iloc[-1]
                        txt_df = pd.DataFrame({"x": [last["TimeStamp"]], "y": [last["Value_Num"]]})
                        txt = alt.Chart(txt_df).mark_text(
                            align="left", dx=6, dy=-6, color=color, fontSize=11, fontWeight="bold"
                        ).encode(x="x:T", y="y:Q", text=alt.value(label))
                        layers.append(txt)

            elif ov_type == "forecast" and df is not None and "Value_Num" in df.columns and "TimeStamp" in df.columns:
                d = df.dropna(subset=["Value_Num", "TimeStamp"]).copy()
                d["TimeStamp"] = pd.to_datetime(d["TimeStamp"], errors="coerce")
                d = d.sort_values("TimeStamp")
                if len(d) >= 5:
                    t0   = d["TimeStamp"].min()
                    secs = (d["TimeStamp"] - t0).dt.total_seconds().values
                    yval = d["Value_Num"].values
                    coeffs = np.polyfit(secs, yval, 1)          # [slope, intercept]
                    periods = max(5, int(ov.get("periods", 20)))
                    intervals = np.diff(secs)
                    step = float(np.median(intervals)) if len(intervals) > 0 else 60.0
                    last_t   = d["TimeStamp"].max()
                    last_sec = secs[-1]
                    future_ts = [last_t + pd.Timedelta(seconds=step * (i + 1)) for i in range(periods)]
                    future_y  = [float(np.polyval(coeffs, last_sec + step * (i + 1))) for i in range(periods)]
                    # Continuity: include last real point
                    anchor = d[["TimeStamp", "Value_Num"]].tail(1)
                    fcst_df = pd.DataFrame({"TimeStamp": future_ts, "Value_Num": future_y})
                    all_df  = pd.concat([anchor, fcst_df], ignore_index=True)
                    fc_line = alt.Chart(all_df).mark_line(
                        color=color, strokeDash=[8, 4], strokeWidth=2, opacity=0.85
                    ).encode(x="TimeStamp:T", y="Value_Num:Q")
                    layers.append(fc_line)
                    # Vertical separator at forecast boundary
                    sep_df = pd.DataFrame({"x": [last_t]})
                    sep = alt.Chart(sep_df).mark_rule(
                        color=color, strokeDash=[4, 4], strokeWidth=1, opacity=0.4
                    ).encode(x="x:T")
                    layers.append(sep)
                    if label:
                        end_df = pd.DataFrame({"x": [future_ts[-1]], "y": [future_y[-1]]})
                        txt = alt.Chart(end_df).mark_text(
                            align="left", dx=6, dy=-6, color=color, fontSize=11, fontWeight="bold"
                        ).encode(x="x:T", y="y:Q", text=alt.value(label))
                        layers.append(txt)

        except Exception:
            continue

    if len(layers) == 1:
        return base_chart
    return alt.layer(*layers).resolve_scale(y="shared")


# ==================================================================
# Modelos predictivos server-side
# ==================================================================
def _holt_smooth(
    values: np.ndarray, alpha: float, beta: float
) -> tuple[np.ndarray, float, float]:
    """Holt double exponential smoothing. Returns (smoothed_array, last_level, last_trend)."""
    alpha = max(0.05, min(0.95, float(alpha)))
    beta  = max(0.01, min(0.50, float(beta)))
    level = float(values[0])
    trend = float(values[1] - values[0]) if len(values) > 1 else 0.0
    smoothed = [level]
    for y in values[1:]:
        prev = level
        level = alpha * float(y) + (1 - alpha) * (level + trend)
        trend = beta  * (level - prev) + (1 - beta) * trend
        smoothed.append(level)
    return np.array(smoothed), level, trend


def _compute_model_series(
    spec: dict, df: pd.DataFrame, horizon_s: float
) -> list[dict]:
    """Ejecuta el modelo predictivo elegido por el LLM y retorna overlays."""
    model  = str(spec.get("model", "holt")).lower()
    params = spec.get("params") or {}

    if df is None or df.empty:
        return []
    if "Value_Num" not in df.columns or "TimeStamp" not in df.columns:
        return []

    d = df.dropna(subset=["Value_Num", "TimeStamp"]).copy()
    d["TimeStamp"] = pd.to_datetime(d["TimeStamp"], errors="coerce")
    d = d.dropna(subset=["TimeStamp"]).sort_values("TimeStamp")
    if len(d) < 5:
        return []

    t0     = d["TimeStamp"].iloc[0]
    t_last = d["TimeStamp"].iloc[-1]
    secs   = (d["TimeStamp"] - t0).dt.total_seconds().values
    yval   = d["Value_Num"].values.astype(float)

    diffs  = np.diff(secs)
    step_s = float(np.median(diffs)) if len(diffs) > 0 else 60.0
    n_pts  = max(30, min(150, int(horizon_s / max(step_s, 1))))
    last_s = secs[-1]
    future_secs = np.array([last_s + step_s * (i + 1) for i in range(n_pts)])
    future_ts   = [t0 + pd.Timedelta(seconds=float(x)) for x in future_secs]

    # ---------- Compute future_y per model ----------
    if model == "polynomial":
        degree = max(2, min(4, int(params.get("degree", 2))))
        try:
            coeffs  = np.polyfit(secs, yval, degree)
            future_y = np.polyval(coeffs, future_secs)
            sigma   = float(np.std(yval - np.polyval(coeffs, secs), ddof=0))
        except Exception:
            coeffs  = np.polyfit(secs, yval, 1)
            future_y = np.polyval(coeffs, future_secs)
            sigma   = float(np.std(yval - np.polyval(coeffs, secs), ddof=0))

    elif model == "holt":
        alpha = float(params.get("alpha", 0.3))
        beta  = float(params.get("beta",  0.1))
        _, level, trend = _holt_smooth(yval, alpha, beta)
        # trend is in units of y per sample; convert to y per second
        avg_interval = float(np.mean(diffs)) if len(diffs) > 0 else step_s
        trend_per_s  = trend / max(avg_interval, 1.0)
        future_y = np.array([level + trend_per_s * (step_s * (i + 1)) for i in range(n_pts)])
        # Residuals from smoothed signal
        smoothed, _, _ = _holt_smooth(yval, alpha, beta)
        sigma = float(np.std(yval - smoothed, ddof=0))

    elif model == "seasonal":
        slope, intercept = np.polyfit(secs, yval, 1)
        residuals = yval - (slope * secs + intercept)
        n = len(residuals)
        period_s = None
        if n >= 8:
            freqs   = np.fft.rfftfreq(n, d=step_s)
            magnitudes = np.abs(np.fft.rfft(residuals))
            if len(magnitudes) > 1:
                dom_idx = int(np.argmax(magnitudes[1:]) + 1)
                dom_freq = freqs[dom_idx]
                period_s = 1.0 / dom_freq if dom_freq > 0 else None

        future_y = []
        periods_back = int(params.get("periods_back", 2))
        for i in range(n_pts):
            x = future_secs[i]
            y_trend = slope * x + intercept
            if period_s and period_s > step_s * 3:
                # Use average residual at same phase from last N periods
                phase = x % period_s
                tol   = step_s * 1.5
                mask  = np.abs(secs % period_s - phase) <= tol
                # Restrict to last periods_back periods
                cutoff_s = last_s - period_s * periods_back
                mask = mask & (secs >= cutoff_s)
                y_seasonal = float(np.mean(residuals[mask])) if mask.any() else 0.0
            else:
                y_seasonal = 0.0
            future_y.append(y_trend + y_seasonal)
        future_y = np.array(future_y)
        sigma = float(np.std(residuals, ddof=0))

    else:  # linear (default)
        coeffs   = np.polyfit(secs, yval, 1)
        future_y = np.polyval(coeffs, future_secs)
        sigma    = float(np.std(yval - np.polyval(coeffs, secs), ddof=0))

    # ---------- Build overlays ----------
    anchor_pt = {"t": t_last.isoformat(), "y": round(float(yval[-1]), 4)}
    data_pts  = [anchor_pt] + [
        {"t": ts.isoformat(), "y": round(float(y), 4)}
        for ts, y in zip(future_ts, future_y)
    ]
    lower_pts = [anchor_pt]
    upper_pts = [anchor_pt]
    for i, (ts, y) in enumerate(zip(future_ts, future_y)):
        widening = 1.0 + (i / max(n_pts, 1)) * 0.6
        ci = 1.96 * sigma * widening
        lower_pts.append({"t": ts.isoformat(), "y": round(float(y - ci), 4)})
        upper_pts.append({"t": ts.isoformat(), "y": round(float(y + ci), 4)})

    return [
        {"type": "series_band", "lower": lower_pts, "upper": upper_pts,
         "color": "#8b5cf6", "opacity": 0.13},
        {"type": "series", "data": data_pts, "color": "#8b5cf6",
         "dashed": True, "width": 2.5, "label": f"Prediccion ({model})"},
        {"type": "vline", "timestamp": t_last.isoformat(),
         "color": "#8b5cf6", "dashed": True, "label": ""},
    ]


def _llm_guided_forecast(
    prompt: str, df: pd.DataFrame, context: dict | None, summary: dict
) -> tuple[list[dict], str]:
    """
    LLM analiza el patron de los datos y elige el mejor modelo.
    Python ejecuta el calculo real. Retorna (overlays, explanation_str).
    """
    horizon_s = _parse_horizon_seconds(prompt)
    if horizon_s is None:
        # Inferir horizonte: mismo span historico, maximo 7 dias
        d = df.dropna(subset=["Value_Num", "TimeStamp"]).copy() if df is not None else pd.DataFrame()
        if not d.empty and "TimeStamp" in d.columns:
            d["TimeStamp"] = pd.to_datetime(d["TimeStamp"], errors="coerce")
            d = d.dropna(subset=["TimeStamp"]).sort_values("TimeStamp")
            if len(d) >= 2:
                span_s = (d["TimeStamp"].iloc[-1] - d["TimeStamp"].iloc[0]).total_seconds()
                horizon_s = min(max(span_s, 3600), 7 * 86400)
        if horizon_s is None:
            horizon_s = 86400

    # Muestra compacta para el LLM (primeros 5, medio 5, ultimos 10)
    recent = summary.get("recent_data", [])
    n = len(recent)
    if n > 20:
        idxs = sorted(set(
            list(range(5)) +
            list(range(n // 2 - 2, n // 2 + 3)) +
            list(range(max(0, n - 10), n))
        ))
        sample = [recent[i] for i in idxs if i < n]
    else:
        sample = recent

    llm_input = {
        "numeric_stats":          summary.get("numeric_stats", {}),
        "sample_interval_seconds": summary.get("sample_interval_seconds"),
        "n_points":                summary.get("n_points"),
        "time_span": {"start": summary.get("time_start"), "end": summary.get("time_end")},
        "sample_values":           sample,
    }

    if horizon_s >= 86400 * 7:
        h_label = f"{horizon_s / (86400*7):.1g} semana(s)"
    elif horizon_s >= 86400:
        h_label = f"{horizon_s / 86400:.1g} dia(s)"
    elif horizon_s >= 3600:
        h_label = f"{horizon_s / 3600:.1g} hora(s)"
    else:
        h_label = f"{horizon_s / 60:.1g} minuto(s)"

    user_msg = (
        f"Peticion: {prompt}\n"
        f"Horizonte de prediccion solicitado: {h_label}\n\n"
        f"Datos:\n{json.dumps(llm_input, ensure_ascii=False, indent=2)}"
    )

    try:
        spec = chat_completion_json(_FORECAST_SYSTEM, user_msg, temperature=0.1)
        if "error" in spec or "model" not in spec:
            spec = {"model": "holt", "params": {"alpha": 0.3, "beta": 0.1},
                    "reasoning": "Modelo Holt por defecto.", "trend_direction": "variable"}
    except Exception:
        spec = {"model": "holt", "params": {"alpha": 0.3, "beta": 0.1},
                "reasoning": "Modelo Holt por defecto.", "trend_direction": "variable"}

    overlays    = _compute_model_series(spec, df, horizon_s)
    model_name  = spec.get("model", "holt")
    reasoning   = spec.get("reasoning", "")
    trend_dir   = spec.get("trend_direction", "")
    explanation = f"Prediccion {h_label} · modelo {model_name} · tendencia: {trend_dir}"
    return overlays, explanation, reasoning


# ==================================================================
# Fragment: renderiza chart + panel IA como unidad aislada
# (st.rerun() solo recarga este bloque, no la pagina entera)
# ==================================================================
@st.fragment
def _chart_ai_fragment(state_key: str, store_key: str) -> None:
    state = st.session_state[state_key]
    store = st.session_state[store_key]
    chart   = store["chart"]
    df      = store["df"]
    context = store["context"]
    height  = store["height"]
    title   = store["title"]

    # -------- Aplicar overlays y renderizar --------
    final_chart = _apply_overlays(chart, state["overlays"], df)
    if height and hasattr(final_chart, "properties"):
        try:
            final_chart = final_chart.properties(height=height)
        except Exception:
            pass

    header_cols = st.columns([8, 1])
    with header_cols[0]:
        if title:
            st.markdown(f"**{title}**")
    with header_cols[1]:
        if st.button(
            "✨ IA" if not state["open"] else "✖ Cerrar",
            key=f"{state_key}_toggle",
            use_container_width=True,
            help="Modo IA: modifica la grafica o hazle preguntas",
        ):
            state["open"] = not state["open"]
            # scope=fragment: solo recarga este bloque, evita scroll al fondo
            st.rerun(scope="fragment")

    st.altair_chart(final_chart, use_container_width=True)

    # -------- Badge de capas activas / boton limpiar contextual --------
    # Leer directo del widget para que el badge sea inmediato sin rerun extra
    current_mode = st.session_state.get(
        f"{state_key}_mode_radio",
        state.get("_mode", "🎨 Modificar grafica"),
    )
    has_overlays = bool(state["overlays"])
    has_history  = bool(state["history"])

    if has_overlays or has_history:
        badge_cols = st.columns([6, 1])
        with badge_cols[0]:
            parts = []
            if has_overlays:
                parts.append(f"🎨 {len(state['overlays'])} modificacion(es)" +
                             (f" — _{state['last_explanation']}_" if state.get("last_explanation") else ""))
            if has_history:
                parts.append(f"💬 {len(state['history'])//2} pregunta(s)")
            st.caption("  ·  ".join(parts))
        with badge_cols[1]:
            # Limpiar segun donde este el usuario
            if current_mode == "🎨 Modificar grafica" and has_overlays:
                if st.button("🗑", key=f"{state_key}_clear", use_container_width=True,
                             help="Limpiar modificaciones de la grafica"):
                    state["overlays"] = []
                    state["last_explanation"] = ""
                    st.rerun(scope="fragment")
            elif current_mode == "💬 Preguntar sobre los datos" and has_history:
                if st.button("🗑", key=f"{state_key}_clear", use_container_width=True,
                             help="Limpiar historial del chat"):
                    state["history"] = []
                    st.rerun(scope="fragment")

    # -------- Panel IA --------
    if not state["open"]:
        return

    with st.container(border=True):
        st.markdown("##### ✨ Modo IA — esta grafica")
        mode_options = ["🎨 Modificar grafica", "💬 Preguntar sobre los datos"]
        default_idx = 1 if state.get("_mode") == "💬 Preguntar sobre los datos" else 0
        mode = st.radio(
            "Accion",
            mode_options,
            index=default_idx,
            horizontal=True,
            key=f"{state_key}_mode_radio",
            label_visibility="collapsed",
        )
        # Guardar modo — el fragment ya re-ejecuta solo cuando cambia el radio
        state["_mode"] = mode

        if mode == "🎨 Modificar grafica":
            _render_modify_panel(state, state_key, df, context)
        else:
            _render_qa_panel(state, state_key, df, context)


# ==================================================================
# Wrapper principal (punto de entrada publico)
# ==================================================================
def chart_with_ai(
    chart: alt.Chart,
    df: pd.DataFrame | None = None,
    chart_id: str = "chart",
    context: dict | None = None,
    title: str | None = None,
    height: int | None = None,
) -> None:
    """
    Renderiza una grafica Altair con un boton de modo IA.
    Usa st.fragment para que las interacciones no recarguen la pagina entera.
    """
    state_key = f"chartai_{chart_id}"
    store_key  = f"chartai_{chart_id}_store"

    if state_key not in st.session_state:
        st.session_state[state_key] = {
            "open": False,
            "overlays": [],
            "history": [],
            "last_explanation": "",
            "_mode": "🎨 Modificar grafica",
        }

    # Guardar chart/df en session_state para que el fragment los acceda en re-runs
    st.session_state[store_key] = {
        "chart": chart, "df": df, "context": context,
        "height": height, "title": title,
    }

    _chart_ai_fragment(state_key, store_key)


# ==================================================================
# Paneles internos (llamados desde el fragment)
# ==================================================================

# Palabras que indican intencion predictiva compleja (LLM-guided)
_COMPLEX_PREDICT_WORDS = (
    "comportamiento", "patron", "modelo pred", "estacional", "ciclico",
    "en base al", "segun los datos", "analiza", "sugier", "recomiend",
    "proximo", "siguiente", "proyect", "extrapolac",
)
# Palabras que indican prediccion simple (local linear, sin LLM)
_SIMPLE_PREDICT_WORDS = (
    "predic", "forecast", "pronostic", "futuro",
)


def _render_modify_panel(state: dict, state_key: str, df, context):
    st.caption("Ejemplos: _\"Dibuja limites a 2 sigma\"_, _\"Agrega media movil de 10 puntos\"_, "
               "_\"Predice el comportamiento 5 dias hacia adelante\"_, _\"Muestra prediccion segun el historial\"_.")

    # st.form: Enter en el campo de texto = clic en Aplicar automaticamente
    with st.form(key=f"{state_key}_mod_form", clear_on_submit=True):
        prompt = st.text_input(
            "Describe que quieres hacer",
            placeholder="Ej: Predice el comportamiento en los proximos 5 dias segun los datos",
            label_visibility="collapsed",
        )
        go = st.form_submit_button("🚀 Aplicar", type="primary", use_container_width=False)

    if go and prompt.strip():
        summary = _build_summary(df, context)
        p_lower = prompt.lower()

        has_ts_data = (
            df is not None and not df.empty
            and "Value_Num" in df.columns
            and "TimeStamp" in df.columns
        )

        # ---- Ruta A: Prediccion compleja con LLM (elige modelo) ----
        is_complex_predict = has_ts_data and (
            any(w in p_lower for w in _COMPLEX_PREDICT_WORDS)
            or (any(w in p_lower for w in _SIMPLE_PREDICT_WORDS) and len(df) >= 20)
        )

        if is_complex_predict:
            with st.spinner("🧠 Analizando patron de datos y calculando modelo predictivo..."):
                overlays, explanation, reasoning = _llm_guided_forecast(prompt, df, context, summary)
            if overlays:
                state["overlays"].extend(overlays)
                state["last_explanation"] = explanation
                st.success(f"\u2713 Modelo predictivo aplicado.")
                with st.expander("📊 Detalle del modelo", expanded=False):
                    st.markdown(f"**{explanation}**")
                    if reasoning:
                        st.markdown(f"*{reasoning}*")
            else:
                st.warning("No se pudo calcular el modelo. La grafica necesita datos numericos con timestamps.")
            st.rerun()
            return

        # ---- Ruta B: Fallback local (rapido, sin LLM) para patrones simples ----
        local = _local_overlay_fallback(prompt, summary, df)
        if local:
            state["overlays"].extend(local)
            state["last_explanation"] = f"{len(local)} capa(s) de control aplicada(s)."
            st.success(f"✓ {len(local)} modificacion(es) aplicada(s).")
            st.rerun()   # full-page: garantiza que la grafica se re-renderice con el overlay
            return

        # ---- Ruta C: LLM libre para overlays visuales ----
        user_msg = (
            f"Peticion del usuario: {prompt}\n\n"
            f"Datos de la grafica (resumen):\n{json.dumps(summary, ensure_ascii=False, indent=2)}"
        )
        try:
            with st.spinner("La IA esta pensando..."):
                result = chat_completion_json(_OVERLAY_SYSTEM, user_msg, temperature=0.1)

            if "error" in result:
                st.error("⚠️ No pude interpretar la respuesta del modelo.")
                with st.expander("Ver respuesta cruda del modelo (debug)"):
                    st.code(result.get("raw", ""), language="text")
                st.info("💡 Reformula con palabras clave: *sigma*, *banda*, *limite*, *promedio*, *maximo*, *minimo*, *tendencia*.")
            else:
                new_overlays = result.get("overlays", []) or []
                explanation  = result.get("explanation", "")
                if not new_overlays:
                    st.warning(explanation or "La IA no propuso modificaciones.")
                else:
                    state["overlays"].extend(new_overlays)
                    state["last_explanation"] = explanation
                    st.success(f"✓ {len(new_overlays)} modificacion(es) aplicada(s). {explanation}")
                    st.rerun()   # full-page: garantiza re-render del chart con overlay
        except Exception as e:
            st.error(f"Error: {e}")


def _render_qa_panel(state: dict, state_key: str, df, context):
    # Mostrar historial de chat
    if state["history"]:
        with st.container(border=False):
            for msg in state["history"][-8:]:
                role = msg["role"]
                if role == "user":
                    st.markdown(f"**🧑 Tu:** {msg['content']}")
                else:
                    st.markdown(f"**🤖 IA:** {msg['content']}")
        st.markdown("---")

    # st.form: Enter = enviar pregunta automaticamente
    with st.form(key=f"{state_key}_qa_form", clear_on_submit=True):
        question = st.text_input(
            "Pregunta sobre esta grafica",
            placeholder="Ej: Que tan estable es esta variable? Hubo anomalias?",
            label_visibility="collapsed",
        )
        ask = st.form_submit_button("💬 Preguntar", type="primary", use_container_width=False)

    if ask and question.strip():
        summary = _build_summary(df, context)
        user_msg = (
            f"Pregunta: {question}\n\n"
            f"Datos de la grafica:\n{json.dumps(summary, ensure_ascii=False, indent=2)}"
        )
        try:
            with st.spinner("Analizando..."):
                answer = chat_completion(_QA_SYSTEM, user_msg, temperature=0.3)
            state["history"].append({"role": "user",      "content": question})
            state["history"].append({"role": "assistant", "content": answer})
            st.rerun()   # full-page: garantiza que el historial se muestre completo
        except Exception as e:
            st.error(f"Error: {e}")

