"""Compare — comparación side-by-side entre máquinas (o mismo parámetro en varias)."""
from datetime import datetime, timedelta, date as _date, time as _time

import streamlit as st
import pandas as pd
import altair as alt

from theme import page_header, COLORS
from api_client import fetch_tag_values
from machine_registry import discover_machines, get_machine_tags
from intent_parser import resolve_time_range
from dashboard_builder import infer_tag_mode
from analytics import numeric_summary


_TYPE_EMOJI = {
    "float":    "🟢",
    "int":      "🔵",
    "bool":     "🟡",
    "string":   "🟠",
    "datetime": "🟣",
    "unknown":  "⚫",
}

def _normalize_type(raw) -> str:
    r = str(raw or "").lower().strip()
    if any(x in r for x in ("double", "float", "single", "real", "number")):
        return "float"
    if any(x in r for x in ("int", "integer", "long", "int32", "int64")):
        return "int"
    if any(x in r for x in ("bool", "boolean", "digital")):
        return "bool"
    if any(x in r for x in ("string", "text", "str")):
        return "string"
    if any(x in r for x in ("time", "date", "timestamp")):
        return "datetime"
    return "unknown"


def render():
    page_header(
        eyebrow="Intelligence · Compare",
        title="Compare",
        subtitle="Compara el mismo parámetro entre varias máquinas en el mismo período.",
    )

    df = discover_machines()
    if df is None or df.empty:
        st.warning("No hay máquinas disponibles.")
        return

    all_names = (df["machine_name"] + "  ·  " + df["line_name"]).tolist()
    BADGE_COLORS = ["#4f46e5", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
    WINDOW_OPTIONS = [
        ("last_hour",  "⚡", "Última hora"),
        ("last_shift", "🔄", "Último turno"),
        ("today",      "📅", "Hoy"),
        ("yesterday",  "📆", "Ayer"),
        ("last_24h",   "🕐", "Últimas 24h"),
        ("last_week",  "📊", "Última semana"),
        ("custom",     "🗓️", "Rango personalizado"),
    ]

    # ── shared vars populated in col_cfg, consumed in col_main ──
    rows_selected   = []
    machine_tags    = {}
    machine_types   = {}
    common_type_map = {}
    common_sorted   = []
    sel_param       = None

    col_cfg, col_main = st.columns([1, 2.5], gap="medium")

    # ══════════════════════════════════════════════════════════════
    # LEFT — configuración
    # ══════════════════════════════════════════════════════════════
    with col_cfg:

        # ── Paso 1 — Máquinas ─────────────────────────────────────
        st.markdown(
            "<div style='background:linear-gradient(135deg,#4f46e5,#7c3aed);"
            "border-radius:12px;padding:12px 16px;margin-bottom:14px;'>"
            "<div style='color:#fff;font-size:0.72rem;font-weight:700;letter-spacing:.1em;"
            "text-transform:uppercase;opacity:.8;'>Paso 1</div>"
            "<div style='color:#fff;font-size:1rem;font-weight:700;'>Máquinas</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        # Init default once so key= binding works immediately (fixes double-click bug)
        if "compare_machines" not in st.session_state:
            st.session_state.compare_machines = all_names[:min(2, len(all_names))]
        st.multiselect(
            "Máquinas",
            options=all_names,
            key="compare_machines",
            placeholder="Elige 2 o más máquinas…",
            label_visibility="collapsed",
        )
        selected = st.session_state.compare_machines

        # Badges de máquinas seleccionadas
        for i, s in enumerate(selected):
            color = BADGE_COLORS[i % len(BADGE_COLORS)]
            name  = s.split("  ·  ")[0]
            st.markdown(
                f"<div style='background:{color}18;border:1px solid {color}44;"
                f"border-radius:8px;padding:6px 10px;margin-bottom:4px;"
                f"font-size:0.8rem;color:{color};font-weight:600;'>"
                f"<span style='display:inline-block;width:8px;height:8px;"
                f"border-radius:50%;background:{color};margin-right:6px;'></span>{name}</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Paso 2 — Período ──────────────────────────────────────
        st.markdown(
            "<div style='background:linear-gradient(135deg,#06b6d4,#0891b2);"
            "border-radius:12px;padding:12px 16px;margin-bottom:14px;'>"
            "<div style='color:#fff;font-size:0.72rem;font-weight:700;letter-spacing:.1em;"
            "text-transform:uppercase;opacity:.8;'>Paso 2</div>"
            "<div style='color:#fff;font-size:1rem;font-weight:700;'>Período</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        window = st.session_state.get("compare_window", "last_hour")
        for wk, wicon, wlabel in WINDOW_OPTIONS:
            if st.button(
                f"{wicon}  {wlabel}",
                key=f"cmp_w_{wk}",
                use_container_width=True,
                type="primary" if window == wk else "secondary",
            ):
                st.session_state.compare_window = wk
                st.rerun()

        # ── Custom date range pickers ──────────────────────────────
        if window == "custom":
            now = datetime.now()
            default_from = now - timedelta(hours=1)

            st.markdown(
                "<div style='margin:10px 0 4px;font-size:0.78rem;font-weight:700;"
                "color:#06b6d4;letter-spacing:.05em;'>DESDE</div>",
                unsafe_allow_html=True,
            )
            cd1, cd2 = st.columns(2)
            with cd1:
                st.date_input(
                    "Fecha inicio", key="cmp_from_date",
                    value=st.session_state.get("cmp_from_date", default_from.date()),
                )
            with cd2:
                st.time_input(
                    "Hora inicio", key="cmp_from_time",
                    value=st.session_state.get("cmp_from_time", _time(default_from.hour, default_from.minute)),
                    step=60,
                )

            st.markdown(
                "<div style='margin:6px 0 4px;font-size:0.78rem;font-weight:700;"
                "color:#10b981;letter-spacing:.05em;'>HASTA</div>",
                unsafe_allow_html=True,
            )
            cd3, cd4 = st.columns(2)
            with cd3:
                st.date_input(
                    "Fecha fin", key="cmp_to_date",
                    value=st.session_state.get("cmp_to_date", now.date()),
                )
            with cd4:
                st.time_input(
                    "Hora fin", key="cmp_to_time",
                    value=st.session_state.get("cmp_to_time", _time(now.hour, now.minute)),
                    step=60,
                )

        st.divider()

        # ── Paso 3 — Parámetro ────────────────────────────────────
        st.markdown(
            "<div style='background:linear-gradient(135deg,#10b981,#059669);"
            "border-radius:12px;padding:12px 16px;margin-bottom:14px;'>"
            "<div style='color:#fff;font-size:0.72rem;font-weight:700;letter-spacing:.1em;"
            "text-transform:uppercase;opacity:.8;'>Paso 3</div>"
            "<div style='color:#fff;font-size:1rem;font-weight:700;'>Parámetro</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        if len(selected) < 2:
            st.markdown(
                "<div style='color:#94a3b8;font-size:0.82rem;padding:8px;'>"
                "Selecciona al menos 2 máquinas primero.</div>",
                unsafe_allow_html=True,
            )
        else:
            rows_selected = [df.iloc[all_names.index(s)] for s in selected]
            for row in rows_selected:
                tags_df = get_machine_tags(row["machine_path"])
                if tags_df.empty:
                    machine_tags[row["machine_name"]]  = {}
                    machine_types[row["machine_name"]] = {}
                    continue
                mapping, types = {}, {}
                for _, tr in tags_df.iterrows():
                    full  = str(tr["piPoint"])
                    short = full.split(".")[-1] if "." in full else full
                    mapping[short] = full
                    types[short]   = _normalize_type(tr.get("type", ""))
                machine_tags[row["machine_name"]]  = mapping
                machine_types[row["machine_name"]] = types

            for types in machine_types.values():
                for short, typ in types.items():
                    if short not in common_type_map:
                        common_type_map[short] = typ

            all_shorts = [set(m.keys()) for m in machine_tags.values()]
            common_sorted = sorted(set.intersection(*all_shorts)) if len(all_shorts) >= 2 else []
            if not common_sorted:
                from collections import Counter
                cnt = Counter()
                for m in machine_tags.values():
                    cnt.update(m.keys())
                common_sorted = sorted(k for k, v in cnt.items() if v >= 2)

            if not common_sorted:
                st.warning("Sin parámetros comunes entre las máquinas seleccionadas.")
            else:
                def fmt_param(short: str) -> str:
                    typ   = common_type_map.get(short, "unknown")
                    emoji = _TYPE_EMOJI.get(typ, "⚫")
                    return f"{emoji}  {short}"

                sel_param = st.selectbox(
                    "Parámetro",
                    options=common_sorted,
                    format_func=fmt_param,
                    key="compare_param_sel",
                    label_visibility="collapsed",
                )
                st.caption(
                    f"📌 {len(common_sorted)} parámetros coincidentes "
                    f"entre las {len(selected)} máquinas"
                )

    # ══════════════════════════════════════════════════════════════
    # RIGHT — resultados
    # ══════════════════════════════════════════════════════════════
    with col_main:

        if len(selected) < 2:
            st.markdown(
                "<div style='display:flex;flex-direction:column;align-items:center;"
                "justify-content:center;padding:80px 20px;"
                "background:#f8fafc;border:2px dashed #cbd5e1;border-radius:16px;margin-top:10px;'>"
                "<div style='font-size:3.5rem;margin-bottom:16px;'>⚖️</div>"
                "<div style='font-size:1.15rem;font-weight:700;color:#0f172a;margin-bottom:8px;'>"
                "Selecciona 2 o más máquinas</div>"
                "<div style='font-size:0.85rem;color:#64748b;text-align:center;max-width:300px;'>"
                "Usa el panel izquierdo para elegir las máquinas que quieres comparar.</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            return

        if not common_sorted or sel_param is None:
            st.info("No se encontraron parámetros comunes entre las máquinas seleccionadas.")
            return

        if window == "custom":
            from_dt = datetime.combine(
                st.session_state.get("cmp_from_date", (datetime.now() - timedelta(hours=1)).date()),
                st.session_state.get("cmp_from_time", _time(0, 0)),
            )
            to_dt = datetime.combine(
                st.session_state.get("cmp_to_date", datetime.now().date()),
                st.session_state.get("cmp_to_time", _time(23, 59)),
            )
            if from_dt >= to_dt:
                st.error("⚠️ La fecha de inicio debe ser anterior a la fecha fin.")
                return
        else:
            from_dt, to_dt = resolve_time_range(window)

        # ── Machine summary cards ──────────────────────────────────
        mcols = st.columns(min(len(rows_selected), 4))
        for i, row in enumerate(rows_selected):
            color  = BADGE_COLORS[i % len(BADGE_COLORS)]
            n_tags = len(machine_tags.get(row["machine_name"], {}))
            mcols[i].markdown(
                f"<div style='background:{color}12;border:1.5px solid {color}55;"
                f"border-radius:12px;padding:12px 10px;text-align:center;'>"
                f"<div style='color:{color};font-size:1.5rem;font-weight:800;'>{n_tags}</div>"
                f"<div style='font-size:0.68rem;color:#64748b;font-weight:600;"
                f"letter-spacing:.05em;'>TAGS</div>"
                f"<div style='font-size:0.78rem;color:#0f172a;font-weight:700;margin-top:4px;"
                f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'"
                f" title='{row['machine_name']}'>{row['machine_name']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Fetch data ─────────────────────────────────────────────
        sources = []
        with st.spinner(f"Cargando «{sel_param}» de {len(rows_selected)} máquinas…"):
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

        numeric_sources = [(m, tn, d) for m, tn, d in sources if infer_tag_mode(d) == "numeric"]
        if not numeric_sources:
            st.info(f"No hay datos numéricos para **{sel_param}** en el período seleccionado.")
            return

        # ── Stats metric cards ─────────────────────────────────────
        srows = []
        for idx, (m, tn, d) in enumerate(numeric_sources):
            s = numeric_summary(d)
            if not s:
                continue
            srows.append({
                "machine": m,
                "color":   BADGE_COLORS[idx % len(BADGE_COLORS)],
                "last":    round(s["last"],   3),
                "mean":    round(s["mean"],   3),
                "std":     round(s["std"],    3),
                "min":     round(s["min"],    3),
                "max":     round(s["max"],    3),
                "cv":      round(s["cv_pct"], 1) if s["cv_pct"] == s["cv_pct"] else None,
            })

        best_idx = max(range(len(srows)), key=lambda i: srows[i]["mean"]) if len(srows) > 1 else 0

        if srows:
            stat_cols = st.columns(len(srows))
            for i, sr in enumerate(srows):
                color = sr["color"]
                star  = " ⭐" if i == best_idx and len(srows) > 1 else ""
                stat_cols[i].markdown(
                    f"<div style='background:{color}10;border:1.5px solid {color}44;"
                    f"border-radius:12px;padding:12px 10px;'>"
                    f"<div style='color:{color};font-size:0.72rem;font-weight:700;"
                    f"text-transform:uppercase;letter-spacing:.06em;overflow:hidden;"
                    f"text-overflow:ellipsis;white-space:nowrap;'>{sr['machine']}{star}</div>"
                    f"<div style='margin-top:8px;'>"
                    f"<div style='display:flex;justify-content:space-between;margin-bottom:3px;'>"
                    f"<span style='font-size:0.73rem;color:#64748b;'>μ media</span>"
                    f"<span style='font-size:0.82rem;font-weight:700;color:#0f172a;'>{sr['mean']}</span></div>"
                    f"<div style='display:flex;justify-content:space-between;margin-bottom:3px;'>"
                    f"<span style='font-size:0.73rem;color:#64748b;'>σ</span>"
                    f"<span style='font-size:0.82rem;font-weight:700;color:#0f172a;'>{sr['std']}</span></div>"
                    f"<div style='display:flex;justify-content:space-between;margin-bottom:3px;'>"
                    f"<span style='font-size:0.73rem;color:#64748b;'>Min</span>"
                    f"<span style='font-size:0.82rem;font-weight:700;color:#0f172a;'>{sr['min']}</span></div>"
                    f"<div style='display:flex;justify-content:space-between;'>"
                    f"<span style='font-size:0.73rem;color:#64748b;'>Max</span>"
                    f"<span style='font-size:0.82rem;font-weight:700;color:#0f172a;'>{sr['max']}</span></div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # ── Overlay chart ──────────────────────────────────────────
        combined = []
        for m, tn, d in numeric_sources:
            sub = d.dropna(subset=["TimeStamp", "Value_Num"])[["TimeStamp", "Value_Num"]].copy()
            sub["Máquina"] = m
            combined.append(sub)
        plot_df = pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()

        if not plot_df.empty:
            domain  = [m for m, _, _ in numeric_sources]
            rng     = BADGE_COLORS[:len(domain)]
            overlay = (
                alt.Chart(plot_df)
                .mark_line(strokeWidth=2.5, opacity=0.9)
                .encode(
                    x=alt.X("TimeStamp:T", title="Tiempo"),
                    y=alt.Y("Value_Num:Q", title=sel_param),
                    color=alt.Color(
                        "Máquina:N",
                        scale=alt.Scale(domain=domain, range=rng),
                        legend=alt.Legend(orient="bottom"),
                    ),
                    tooltip=[
                        "Máquina:N", "TimeStamp:T",
                        alt.Tooltip("Value_Num:Q", format=".3f", title=sel_param),
                    ],
                )
                .properties(height=380, title=f"Overlay — {sel_param}")
                .interactive()
            )
            with st.container(border=True):
                st.altair_chart(overlay, use_container_width=True)

        # ── Stats table ────────────────────────────────────────────
        if srows:
            table_data = [{
                "Máquina": sr["machine"],
                "Último":  sr["last"],
                "μ media": sr["mean"],
                "σ":       sr["std"],
                "Min":     sr["min"],
                "Max":     sr["max"],
                "CV%":     sr["cv"],
            } for sr in srows]
            with st.container(border=True):
                st.markdown(f"##### 📋 Estadísticas comparativas — {sel_param}")
                st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


