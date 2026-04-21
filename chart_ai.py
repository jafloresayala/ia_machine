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
import pandas as pd
import altair as alt
import streamlit as st

from llm_engine import chat_completion_json, chat_completion


# ==================================================================
# Sistema prompts
# ==================================================================
_OVERLAY_SYSTEM = """Eres un asistente que modifica graficas Altair. Responde SIEMPRE con JSON valido.

Esquema exacto (responde SOLO con este objeto, sin texto adicional):
{
  "overlays": [
    { "type": "hline", "value": 123.45, "label": "Mean", "color": "#10b981", "dashed": false },
    { "type": "band",  "lower": 100.0, "upper": 150.0, "label": "Control +/-1sigma", "color": "#f59e0b", "opacity": 0.15 },
    { "type": "vline", "timestamp": "2026-04-20T08:30:00", "label": "Evento", "color": "#ef4444", "dashed": true },
    { "type": "trendline", "color": "#3b82f6", "label": "Tendencia" }
  ],
  "explanation": "texto breve de lo que hiciste"
}

REGLAS CRITICAS:
- SIEMPRE calcula numeros reales. Nunca pongas formulas como "mean + 2*std". Haz la aritmetica tu mismo con los stats provistos.
- Usa los valores exactos del campo "numeric_stats" del usuario (mean, std, min, max, median).
- Para "bandas +/- N sigma" genera UN objeto band con lower=mean-N*std y upper=mean+N*std.
- Para "limites upper/lower +/- N sigma" genera DOS objetos hline (uno con mean+N*std, otro con mean-N*std).
- Puedes combinar varios overlays en un solo array.

EJEMPLOS (si mean=100, std=5):

Usuario: "Dibuja limites upper y lower a 2 desviaciones estandar"
Respuesta:
{"overlays":[
 {"type":"hline","value":110.0,"label":"Upper +2sigma","color":"#ef4444","dashed":true},
 {"type":"hline","value":90.0,"label":"Lower -2sigma","color":"#ef4444","dashed":true},
 {"type":"hline","value":100.0,"label":"Mean","color":"#10b981","dashed":false}
],"explanation":"Agregue limites de control a +/- 2 sigma del promedio"}

Usuario: "Agrega bandas de control +/- sigma 1.5"
Respuesta:
{"overlays":[
 {"type":"band","lower":92.5,"upper":107.5,"label":"Control +/-1.5sigma","color":"#f59e0b","opacity":0.15}
],"explanation":"Banda de control a +/- 1.5 sigma del promedio"}

Usuario: "Marca el maximo y agrega linea de tendencia"
Respuesta:
{"overlays":[
 {"type":"hline","value":<max_real>,"label":"Maximo","color":"#ef4444","dashed":false},
 {"type":"trendline","color":"#3b82f6","label":"Tendencia"}
],"explanation":"Marca el maximo y dibuja tendencia lineal"}

Si no puedes cumplir, responde {"overlays":[],"explanation":"razon"}.
NUNCA incluyas texto fuera del JSON."""

_QA_SYSTEM = """Eres un analista de datos industriales SMT. Responde preguntas sobre la grafica/datos.
Se conciso y tecnico. Usa los stats y muestras provistos. Responde en espanol."""


# ==================================================================
# Construccion de overlays
# ==================================================================
def _local_overlay_fallback(prompt: str, summary: dict) -> list[dict] | None:
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

    return overlays if overlays else None


def _build_summary(df: pd.DataFrame | None, context: dict | None) -> dict:
    """Resumen compacto de estadisticas para enviar al LLM."""
    summary = dict(context or {})
    if df is None or df.empty:
        return summary

    # Rango temporal
    if "TimeStamp" in df.columns:
        ts = pd.to_datetime(df["TimeStamp"], errors="coerce").dropna()
        if not ts.empty:
            summary["time_start"] = ts.min().isoformat()
            summary["time_end"] = ts.max().isoformat()
            summary["n_points"] = int(len(ts))

    # Stats numericos
    if "Value_Num" in df.columns:
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
            }

    # Categoricos
    if "Value_Str" in df.columns and "Value_Num" not in df.columns:
        v = df["Value_Str"].dropna().astype(str)
        if not v.empty:
            top = v.value_counts().head(5).to_dict()
            summary["categorical_top"] = {str(k): int(v) for k, v in top.items()}
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

        except Exception:
            continue

    if len(layers) == 1:
        return base_chart
    return alt.layer(*layers).resolve_scale(y="shared")


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
            st.rerun()

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
                    st.rerun()
            elif current_mode == "💬 Preguntar sobre los datos" and has_history:
                if st.button("🗑", key=f"{state_key}_clear", use_container_width=True,
                             help="Limpiar historial del chat"):
                    state["history"] = []
                    st.rerun()

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
def _render_modify_panel(state: dict, state_key: str, df, context):
    st.caption("Ejemplos: _\"Dibuja limites superior e inferior a 2 desviaciones estandar\"_, "
               "_\"Marca el valor promedio\"_, _\"Agrega una linea de tendencia\"_.")

    # st.form: Enter en el campo de texto = clic en Aplicar automaticamente
    with st.form(key=f"{state_key}_mod_form", clear_on_submit=True):
        prompt = st.text_input(
            "Describe que quieres hacer",
            placeholder="Ej: Agrega bandas de control a +/- 2 sigma del promedio",
            label_visibility="collapsed",
        )
        go = st.form_submit_button("🚀 Aplicar", type="primary", use_container_width=False)

    if go and prompt.strip():
        summary = _build_summary(df, context)

        # 1) Fallback local (rapido, sin LLM) para patrones comunes
        local = _local_overlay_fallback(prompt, summary)
        if local:
            state["overlays"].extend(local)
            state["last_explanation"] = f"{len(local)} capa(s) de control aplicada(s)."
            st.success(f"✓ {len(local)} modificacion(es) aplicada(s).")
            st.rerun()
            return

        # 2) Delegar al LLM
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
                    st.rerun()
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
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

