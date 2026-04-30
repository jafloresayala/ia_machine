"""Live Monitor — KPIs, health score y gráficas en tiempo real de una máquina."""
from datetime import datetime, timedelta, date, time
import io

import streamlit as st
import pandas as pd

# Import opcional de fpdf2
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

from theme import page_header, COLORS
from api_client import fetch_tag_values
from machine_registry import discover_machines, get_machine_tags, find_machine
from dashboard_builder import render_machine_dashboard, infer_tag_mode
from analytics import numeric_summary, categorical_summary


def _build_pdf(summary_df: pd.DataFrame, machine_name: str, time_label: str) -> bytes:
    """Genera un PDF estético del snapshot de parámetros usando fpdf2."""
    if not FPDF_AVAILABLE:
        st.error("❌ fpdf2 no está instalado. Instala con: pip install fpdf2")
        return b""

    def _latin(text: str) -> str:
        return str(text).encode("latin-1", "replace").decode("latin-1")

    class SnapshotPDF(FPDF):
        def header(self):
            self.set_fill_color(99, 102, 241)
            self.rect(0, 0, 297, 22, "F")
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(255, 255, 255)
            self.set_xy(12, 5)
            self.cell(0, 8, "Machine Intelligence Platform", align="L")
            self.set_font("Helvetica", "", 8)
            self.set_xy(12, 14)
            self.cell(0, 5, "Parametros - Snapshot Report", align="L")
            self.set_text_color(0, 0, 0)

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(148, 163, 184)
            now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            self.cell(
                0, 5,
                f"MIP - Machine Intelligence Platform  |  {now_str}  |  Pag {self.page_no()}",
                align="C",
            )

    pdf = SnapshotPDF(orientation="L", unit="mm", format="A4")
    pdf.set_margins(12, 26, 12)
    # Auto page break disabled — we manage breaks manually to avoid mid-row splits
    pdf.set_auto_page_break(auto=False)
    _BOTTOM_MARGIN = 14  # mm
    pdf.add_page()

    # Info block
    pdf.set_y(26)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 6, _latin(f"Maquina: {machine_name}"), ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, _latin(f"Rango: {time_label}"), ln=True)
    pdf.cell(0, 5, _latin(f"Generado: {datetime.now().strftime('%d/%m/%Y  %H:%M:%S')}"), ln=True)
    pdf.ln(3)

    # Separator line
    pdf.set_draw_color(99, 102, 241)
    pdf.set_line_width(0.5)
    pdf.line(12, pdf.get_y(), 285, pdf.get_y())
    pdf.ln(4)

    cols = list(summary_df.columns)

    # Anchos proporcionales: 273mm disponibles (A4 landscape - márgenes 12mm c/lado)
    TOTAL_W = 273.0
    weight_map = {
        "parametro": 5, "tipo": 2, "timestamp": 3,
        "ultimo valor": 2.5, "min": 2, "max": 2,
        "promedio": 2, "cv%": 1.5, "outliers": 2,
    }

    def _norm(col: str) -> str:
        return (col.lower()
                .replace("á","a").replace("é","e").replace("í","i")
                .replace("ó","o").replace("ú","u").replace("ü","u"))

    weights = [weight_map.get(_norm(c), 2.0) for c in cols]
    total_w = sum(weights)
    col_widths = [round(TOTAL_W * w / total_w, 1) for w in weights]

    align_map = {"parametro": "L", "tipo": "C", "timestamp": "C"}
    col_aligns = [align_map.get(_norm(c), "C") for c in cols]

    LINE_H = 5.5   # altura de cada línea de texto dentro de la celda
    PAD_H  = 2.0   # padding vertical extra por celda

    from fpdf.enums import XPos, YPos

    def _row_height(row_vals: list[str], font_style: str) -> float:
        """Calcula la altura necesaria para la fila más alta (dry_run)."""
        max_lines = 1
        for val, w in zip(row_vals, col_widths):
            pdf.set_font("Helvetica", font_style, 8)
            lines = pdf.multi_cell(w, LINE_H, val, dry_run=True, output="LINES")
            max_lines = max(max_lines, len(lines))
        return max_lines * LINE_H + PAD_H

    def _draw_row(row_vals: list[str], row_h: float, fill_rgb: tuple,
                  text_rgb: tuple, font_style: str):
        """Dibuja una fila completa con multi_cell wrappable."""
        y_top = pdf.get_y()
        pdf.set_x(pdf.l_margin)
        pdf.set_fill_color(*fill_rgb)
        pdf.set_text_color(*text_rgb)
        pdf.set_font("Helvetica", font_style, 8)
        for val, w, align in zip(row_vals, col_widths, col_aligns):
            pdf.multi_cell(
                w, LINE_H, val,
                fill=True, align=align,
                max_line_height=LINE_H,
                new_x=XPos.RIGHT, new_y=YPos.TOP,
            )
        pdf.set_y(y_top + row_h)
        # Separador horizontal sutil
        pdf.set_draw_color(226, 232, 240)
        pdf.set_line_width(0.1)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + TOTAL_W, pdf.get_y())

    # ---- Header ----
    header_vals = [_latin(c) for c in cols]
    h_h = _row_height(header_vals, "B")
    _draw_row(header_vals, h_h, (99, 102, 241), (255, 255, 255), "B")

    # ---- Data rows ----
    for i, (_, row_data) in enumerate(summary_df.iterrows()):
        row_vals = [_latin(str(row_data[c])) for c in cols]
        row_h = _row_height(row_vals, "")
        # Manual page break: if row doesn't fit, start a new page and repeat header
        if pdf.get_y() + row_h > pdf.h - _BOTTOM_MARGIN:
            pdf.add_page()
            _draw_row(header_vals, h_h, (99, 102, 241), (255, 255, 255), "B")
        bg = (248, 250, 252) if i % 2 == 0 else (255, 255, 255)
        _draw_row(row_vals, row_h, bg, (30, 41, 59), "")

    return bytes(pdf.output())


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

    # ---- CV% promedio ----
    cv_values = []
    for tn, df in tag_data.items():
        s_tmp = numeric_summary(df)
        if s_tmp and not (s_tmp["cv_pct"] != s_tmp["cv_pct"]):  # excluir NaN
            cv_values.append(s_tmp["cv_pct"])
    avg_cv = float(sum(cv_values) / len(cv_values)) if cv_values else None

    kc = st.columns(5)
    with kc[0]: st.metric("Máquina", name)
    with kc[1]: st.metric("Rango", time_label[:30] + ("…" if len(time_label) > 30 else ""))
    with kc[2]: st.metric("Tags activos", len(tag_data))
    with kc[3]: st.metric("Puntos", f"{len(all_data):,}")
    with kc[4]:
        if avg_cv is not None:
            st.metric("CV% promedio", f"{avg_cv:.1f}%")
        else:
            st.metric("CV% promedio", "N/A")

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ---- Resumen de tags estilo Databricks ----
    with st.container(border=True):
        hcol, tcol = st.columns([5, 2])
        with hcol:
            st.markdown("##### 📊 Parámetros — snapshot")
        with tcol:
            view_options = ["📋 Tabla"]
            if FPDF_AVAILABLE:
                view_options.append("📄 PDF")
            snap_view = st.radio(
                "Vista",
                view_options,
                horizontal=True,
                label_visibility="collapsed",
                key="monitor_snap_view",
            )

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
                })
            else:
                s = categorical_summary(df)
                rows.append({
                    "Parámetro": short, "Tipo": "state",
                    "Timestamp": last_ts,
                    "Último valor": s.get("last", "—") if s else "—",
                    "Mín": "—", "Máx": "—", "Promedio": "—", "CV%": "—",
                    "Outliers": "—",
                })
        summary_df = pd.DataFrame(rows)

        if snap_view == "📋 Tabla":
            st.dataframe(summary_df, use_container_width=True, hide_index=True, height=min(420, 80 + 36 * len(rows)))

            # Download como Excel
            xl_buf = io.BytesIO()
            with pd.ExcelWriter(xl_buf, engine="openpyxl") as writer:
                summary_df.to_excel(writer, index=False, sheet_name="Snapshot")
            st.download_button(
                label="⬇️ Descargar tabla (Excel)",
                data=xl_buf.getvalue(),
                file_name=f"snapshot_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        else:
            # ---- Vista PDF (HTML estético) ----
            gen_time = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
            header_row = "".join(
                f"<th>{c}</th>" for c in summary_df.columns
            )
            body_rows = ""
            for i, row_data in summary_df.iterrows():
                tipo = row_data.get("Tipo", "")
                tipo_badge = (
                    "<span style='background:#6366f1;color:#fff;padding:1px 7px;"
                    "border-radius:10px;font-size:0.7rem;'>numeric</span>"
                    if tipo == "numeric"
                    else
                    "<span style='background:#06b6d4;color:#fff;padding:1px 7px;"
                    "border-radius:10px;font-size:0.7rem;'>state</span>"
                )
                cells = []
                for col in summary_df.columns:
                    val = row_data[col]
                    if col == "Tipo":
                        cells.append(f"<td style='text-align:center'>{tipo_badge}</td>")
                    elif col == "Parámetro":
                        cells.append(f"<td style='font-weight:600;color:#1e293b'>{val}</td>")
                    else:
                        cells.append(f"<td style='text-align:center'>{val}</td>")
                row_bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
                body_rows += f"<tr style='background:{row_bg}'>{''.join(cells)}</tr>"

            html_report = f"""
            <div style='font-family:Inter,Segoe UI,sans-serif;background:#ffffff;
                        border-radius:12px;padding:28px 32px;
                        box-shadow:0 2px 16px rgba(0,0,0,.08);'>
              <!-- Header -->
              <div style='display:flex;justify-content:space-between;align-items:flex-start;
                          border-bottom:2px solid #6366f1;padding-bottom:14px;margin-bottom:18px;'>
                <div>
                  <div style='font-size:1.35rem;font-weight:700;color:#1e293b;letter-spacing:-.02em;'>
                    Machine Intelligence Platform
                  </div>
                  <div style='font-size:0.82rem;color:#64748b;margin-top:2px;'>
                    Parámetros — Snapshot Report
                  </div>
                </div>
                <div style='text-align:right;font-size:0.78rem;color:#64748b;line-height:1.6;'>
                  <div><b>Máquina:</b> {name}</div>
                  <div><b>Rango:</b> {time_label}</div>
                  <div><b>Generado:</b> {gen_time}</div>
                </div>
              </div>
              <!-- Tabla -->
              <table style='width:100%;border-collapse:collapse;font-size:0.82rem;color:#1e293b;'>
                <thead>
                  <tr style='background:#6366f1;color:#ffffff;'>
                    {header_row}
                  </tr>
                </thead>
                <tbody>
                  {body_rows}
                </tbody>
              </table>
              <!-- Footer -->
              <div style='margin-top:18px;padding-top:10px;border-top:1px solid #e2e8f0;
                          font-size:0.72rem;color:#94a3b8;display:flex;justify-content:space-between;'>
                <span>MIP — Machine Intelligence Platform</span>
                <span>{len(rows)} parámetros  ·  {gen_time}</span>
              </div>
            </div>
            """
            st.markdown(html_report, unsafe_allow_html=True)

            # Download como PDF real
            pdf_bytes = _build_pdf(summary_df, name, time_label)
            st.download_button(
                label="⬇️ Descargar reporte (PDF)",
                data=pdf_bytes,
                file_name=f"snapshot_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    # ---- Dashboard completo ----
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        render_machine_dashboard(name, tag_data, time_range_label=time_label)

    # Botón refresh
    if st.button("🔄 Actualizar datos", type="primary", use_container_width=True):
        st.rerun()
