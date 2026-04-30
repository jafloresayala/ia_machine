"""Explorer — navegación de la jerarquía PI: líneas → máquinas → parámetros."""
import streamlit as st
import pandas as pd
from datetime import datetime

from theme import page_header, COLORS
from state import goto
from machine_registry import discover_machines, get_machine_tags, get_catalog_info


def render():
    page_header(
        eyebrow="Data · Catalog",
        title="Explorer",
        subtitle="Navega por todas las líneas, máquinas y parámetros disponibles en PI.",
    )

    # -------- Catálogo con caché persistente --------
    # 1° prioridad: session state (instantáneo, no lee ni disco ni red)
    # 2° prioridad: disco (archivo JSON, sobrevive reinicios)
    # 3° prioridad: descubrir desde PI (lento, solo la primera vez)
    if "explorer_catalog" not in st.session_state:
        st.session_state.explorer_catalog = discover_machines()

    df = st.session_state.explorer_catalog

    # -------- Barra de estado + botón Actualizar --------
    info = get_catalog_info()
    col_info, col_btn = st.columns([5, 1])
    with col_info:
        if info["saved_at"]:
            delta = datetime.now() - info["saved_at"]
            total_min = int(delta.total_seconds()) // 60
            h, m = divmod(total_min, 60)
            age_str = f"{h}h {m}m" if h else f"{m}m"
            st.caption(f"🗄 Catálogo: **{info['count']} máquinas** · guardado hace {age_str}")
        else:
            st.caption(f"🗄 Catálogo: **{len(df)} máquinas**")
    with col_btn:
        if st.button("🔄 Actualizar", use_container_width=True,
                     help="Re-escanea la jerarquía PI y actualiza el catálogo"):
            with st.spinner("Actualizando catálogo…"):
                new_df = discover_machines(force_refresh=True)
            added = len(new_df) - len(df)
            st.session_state.explorer_catalog = new_df
            # Limpiar tags cacheados en session state al refrescar
            st.session_state.pop("explorer_tags", None)
            df = new_df
            if added > 0:
                st.success(f"Catálogo actualizado — {added} máquina(s) nueva(s) encontradas.")
            elif added < 0:
                st.success(f"Catálogo actualizado — {abs(added)} máquina(s) eliminadas.")
            else:
                st.success("Catálogo actualizado — sin cambios.")
            st.rerun()

    if df is None or df.empty:
        st.warning("No se pudo cargar el catálogo de máquinas. Revisa la conexión con PI.")
        return

    # -------- Filtros --------
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            q = st.text_input("🔎 Buscar máquina", value=st.session_state.get("explorer_query", ""),
                              placeholder="Ej: Paste Printer, Reflow, SPI, L1L…")
            st.session_state.explorer_query = q
        with c2:
            lines = ["(Todas las líneas)"] + sorted(df["line_name"].unique().tolist())
            sel_line = st.selectbox("📍 Línea", lines)
        with c3:
            view_mode = st.radio("Vista", ["Cards", "Tabla"], horizontal=True, label_visibility="collapsed")

    filtered = df.copy()
    if q:
        mask = (
            filtered["machine_name"].str.contains(q, case=False, na=False) |
            filtered["line_name"].str.contains(q, case=False, na=False)
        )
        filtered = filtered[mask]
    if sel_line != "(Todas las líneas)":
        filtered = filtered[filtered["line_name"] == sel_line]

    # -------- KPIs --------
    kc = st.columns(4)
    with kc[0]: st.metric("Total", len(df))
    with kc[1]: st.metric("Coincidencias", len(filtered))
    with kc[2]: st.metric("Líneas", filtered["line_name"].nunique() if not filtered.empty else 0)
    with kc[3]: st.metric("Máquinas únicas", filtered["machine_name"].nunique() if not filtered.empty else 0)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    if filtered.empty:
        st.info("Sin resultados con esos filtros.")
        return

    # -------- Render --------
    if view_mode == "Tabla":
        st.dataframe(
            filtered[["line_name", "machine_name", "machine_path"]],
            use_container_width=True, hide_index=True, height=520,
            column_config={
                "line_name":   st.column_config.TextColumn("Línea", width="medium"),
                "machine_name":st.column_config.TextColumn("Máquina"),
                "machine_path":st.column_config.TextColumn("Path PI"),
            },
        )
        return

    # Cards
    for line_name, group in filtered.groupby("line_name"):
        st.markdown(f"##### 📍 {line_name}  &nbsp; <span class='pill pill-neutral'>{len(group)} máquinas</span>",
                    unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (_, row) in enumerate(group.iterrows()):
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.4rem;">
                      <div style="width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#4f46e5,#06b6d4);color:#fff;display:flex;align-items:center;justify-content:center;font-size:15px;font-weight:700;">
                        {row['machine_name'][:2].upper()}
                      </div>
                      <div style="font-weight:600;color:{COLORS['ink']};line-height:1.15;">
                        {row['machine_name']}
                      </div>
                    </div>
                    <div class="mono" style="color:{COLORS['ink_muted']};font-size:0.72rem;word-break:break-all;margin-bottom:0.5rem;">
                      {row['machine_path']}
                    </div>
                    """, unsafe_allow_html=True)
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("📡 Monitor", key=f"exp_mon_{line_name}_{row['machine_name']}_{i}", use_container_width=True):
                            st.session_state.selected_machine_name = row["machine_name"]
                            st.session_state.selected_machine_path = row["machine_path"]
                            goto("monitor")
                    with b2:
                        if st.button("🎮 Playground", key=f"exp_an_{line_name}_{row['machine_name']}_{i}", use_container_width=True):
                            st.session_state.selected_machine_name = row["machine_name"]
                            st.session_state.selected_machine_path = row["machine_path"]
                            st.session_state.analytics_machine = row["machine_name"]
                            st.session_state.playground_params = []  # reset param selection for the new machine
                            goto("analytics")

                    with st.expander("Ver tags"):
                        # Tags se cargan bajo demanda y se guardan en session state
                        if "explorer_tags" not in st.session_state:
                            st.session_state.explorer_tags = {}
                        path = row["machine_path"]
                        if path not in st.session_state.explorer_tags:
                            if st.button("📋 Cargar tags", key=f"load_tags_{line_name}_{i}",
                                         use_container_width=True):
                                with st.spinner("Cargando…"):
                                    st.session_state.explorer_tags[path] = get_machine_tags(path)
                                st.rerun()
                        else:
                            tags = st.session_state.explorer_tags[path]
                            if tags.empty:
                                st.caption("Sin tags disponibles.")
                            else:
                                st.caption(f"{len(tags)} tags")
                                st.dataframe(tags, use_container_width=True, hide_index=True, height=180)
