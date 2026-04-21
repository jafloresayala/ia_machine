"""Vista Agent — interfaz conversacional con el agente de IA."""
import io
from datetime import datetime, date, time, timedelta

import pandas as pd
import streamlit as st

from theme import page_header, COLORS
from config import QUERY_SUGGESTIONS, OLLAMA_MODEL
from api_client import fetch_tag_values
from machine_registry import (
    discover_machines, find_machine, smart_find_machine, get_machine_tags, get_machine_names,
)
from llm_engine import answer_user_question, chat_as_agent, chat_as_agent_stream, llm_pick_machine
from intent_parser import parse_query, parse_query_fallback, resolve_time_range
from dashboard_builder import (
    render_machine_dashboard, render_machine_list, build_data_summary_text,
)


# ================================================================
# Core logic (migrated from old app.py)
# ================================================================
def _execute_intent(intent):
    result = {"type": "text", "content": "", "dashboard_data": None}

    if intent.action == "unknown" or intent.error:
        # Si Ollama está disponible, dejar que el LLM responda libremente
        # en lugar de mostrar un mensaje de error estático
        if st.session_state.ollama_ok:
            try:
                reply = chat_as_agent(intent.raw_query, machine_names=get_machine_names())
                result["content"] = reply
            except Exception:
                result["content"] = (
                    intent.error
                    or "No pude procesar tu pregunta. Intenta algo como:\n"
                    "- *¿Cuáles son los datos de hoy de la Paste Printer L1L?*\n"
                    "- *Muéstrame outliers del Reflow de la Línea 3*"
                )
        else:
            result["content"] = (
                intent.error
                or "No entendí tu pregunta. Prueba algo como:\n"
                "- *¿Cuáles son los datos de hoy de la Paste Printer L1L?*\n"
                "- *Muéstrame outliers del Reflow de la Línea 3*"
            )
        return result

    if intent.action == "chat":
        try:
            reply = chat_as_agent(intent.raw_query, machine_names=get_machine_names())
            result["content"] = reply
        except Exception as e:
            result["content"] = "Hubo un error al conectar con el modelo de IA."
        return result

    if intent.action == "list_machines":
        result["type"] = "machine_list"
        result["content"] = "Aquí están las máquinas disponibles:"
        return result

    if intent.action in ("show_dashboard", "show_attribute", "status_check", "compare"):
        raw_query = intent.raw_query or ""

        # ---- Resolver maquina con busqueda inteligente ----
        if not intent.machine_name:
            # El parser no extrajo nombre: buscar por el texto completo del query
            matches = smart_find_machine(raw_query, line_hint=intent.line_hint)
        else:
            matches = find_machine(intent.machine_name, line_hint=intent.line_hint)
            if matches.empty:
                # Busqueda ampliada con sinonimos
                matches = smart_find_machine(intent.machine_name, line_hint=intent.line_hint)

        # ---- Fallback: LLM elige entre todos los candidatos ----
        if matches.empty and st.session_state.ollama_ok:
            all_names = get_machine_names()
            with st.spinner("Buscando la maquina mas relevante..."):
                chosen = llm_pick_machine(raw_query, all_names)
            if chosen:
                matches = find_machine(chosen, line_hint=intent.line_hint)
                if matches.empty:
                    matches = smart_find_machine(chosen, line_hint=intent.line_hint)

        if matches.empty:
            all_names = get_machine_names()
            # Mostrar solo los primeros de cada tipo para no saturar
            result["content"] = (
                "No encontre una maquina que coincida con tu descripcion. "
                "Puedes buscar por tipo (Reflow, Paste Printer, AOI, SPI...) o por linea. "
                "Maquinas disponibles:\n"
                + ", ".join(f"**{n}**" for n in all_names[:20])
                + (f" ...y {len(all_names)-20} mas." if len(all_names) > 20 else "")
            )
            return result

        if len(matches) > 1:
            result["type"] = "disambiguation"
            result["matches"] = matches
            result["intent"] = intent
            machine_list = "\n".join(
                f"- **{row['machine_name']}** · {row['line_name']}"
                for _, row in matches.iterrows()
            )
            result["content"] = (
                f"Encontré **{len(matches)}** máquinas que coinciden con "
                f"**{intent.machine_name}**. ¿A cuál te refieres?\n\n{machine_list}"
            )
            st.session_state.pending_intent = intent
            return result

        machine = matches.iloc[0]
        machine_path = machine["machine_path"]
        machine_name = machine["machine_name"]

        tags_df = get_machine_tags(machine_path)
        if tags_df.empty:
            result["content"] = f"La máquina **{machine_name}** no tiene tags consultables."
            return result

        if intent.attribute_name and intent.action == "show_attribute":
            attr_lower = intent.attribute_name.lower()
            mask = tags_df["name"].str.lower().str.contains(attr_lower, na=False)
            filtered = tags_df[mask]
            if not filtered.empty:
                tags_df = filtered

        from_dt = intent.from_dt or datetime.now().replace(hour=0, minute=0, second=0)
        to_dt = intent.to_dt or datetime.now()
        tag_names = tags_df["piPoint"].astype(str).tolist()  # sin limite

        with st.spinner(f"Consultando {len(tag_names)} tags de {machine_name}…"):
            all_data = fetch_tag_values(tag_names, from_dt, to_dt)

        if all_data.empty:
            result["content"] = f"No hay datos disponibles para **{machine_name}** en el rango."
            return result

        tag_data = {}
        for tag_name in tag_names:
            tag_df = all_data[all_data["Tag_Name"] == tag_name].copy()
            if not tag_df.empty:
                tag_data[tag_name] = tag_df

        # Determinar qué herramientas mostrar
        show_dashboard = st.session_state.opt_dashboard
        show_excel     = st.session_state.opt_excel
        show_analysis  = st.session_state.opt_analysis

        # Si NINGUNA herramienta está activa, la IA decide según el prompt
        if not show_dashboard and not show_excel and not show_analysis:
            q = (intent.raw_query or "").lower()
            # Palabras clave que indican análisis / tendencias
            analysis_kw = {"analiz", "tendencia", "trend", "outlier", "anomal", "variab",
                           "compara", "estable", "inestable", "fluctu", "evalúa", "evalua",
                           "resume", "resum", "explica", "explica", "diagnos"}
            # Palabras clave que indican exportación
            export_kw   = {"excel", "exporta", "descarga", "csv", "archivo", "report"}
            # Palabras clave que piden gráfica explícita
            chart_kw    = {"gráfica", "grafica", "chart", "visual", "dashboard",
                           "muestra", "pinta", "plot", "dibuja"}

            wants_analysis = any(k in q for k in analysis_kw)
            wants_export   = any(k in q for k in export_kw)
            wants_chart    = bool(intent.chart_instructions) or any(k in q for k in chart_kw)

            # Por defecto siempre mostrar dashboard; análisis si IA disponible
            show_dashboard = wants_chart or (not wants_analysis and not wants_export)
            show_analysis  = wants_analysis or (st.session_state.ollama_ok and not wants_export)
            show_excel     = wants_export

        result.update({
            "type":  "dashboard",
            "machine_name":   machine_name,
            "tag_data":       tag_data,
            "all_data":       all_data,
            "time_range":     intent.time_range,
            "chart_instructions": intent.chart_instructions or [],
            "content": intent.summary_text or f"Datos de {machine_name} ({intent.time_range})",
            "show_dashboard": show_dashboard,
            "show_excel":     show_excel,
            "show_analysis":  show_analysis,
        })

        if show_analysis and st.session_state.ollama_ok and tag_data:
            try:
                summary_text = build_data_summary_text(tag_data)
                with st.spinner("🧠 Generando análisis inteligente…"):
                    llm_answer = answer_user_question(
                        user_question=intent.raw_query,
                        machine_name=machine_name,
                        data_summary=summary_text,
                    )
                result["llm_summary"] = llm_answer
            except Exception:
                result["llm_summary"] = None

        return result

    result["content"] = "Procesando tu solicitud…"
    return result


def _process_query(user_query: str):
    """Agrega el mensaje del usuario al historial y marca para procesamiento diferido."""
    st.session_state.chat_history.append({
        "role": "user", "content": user_query, "timestamp": datetime.now(),
    })
    st.session_state._pending_query = user_query


def _process_pending():
    """
    Procesa la consulta pendiente con streaming para respuestas conversacionales.
    Llamado en render() despues de mostrar el historial, para que el usuario
    vea su mensaje de inmediato antes de esperar la respuesta.
    """
    query = st.session_state.pop("_pending_query")
    time_override = st.session_state.get("ui_time_range", "today")
    custom_from = st.session_state.get("ui_custom_from", date.today())
    custom_to   = st.session_state.get("ui_custom_to",   date.today())

    # Parsear intención
    if st.session_state.ollama_ok:
        with st.spinner("Interpretando..."):
            try:
                intent = parse_query(query)
            except Exception:
                intent = parse_query_fallback(query, get_machine_names())
    else:
        intent = parse_query_fallback(query, get_machine_names())

    # Aplicar override de tiempo
    if time_override == "custom":
        intent.time_range = "custom"
        intent.from_dt = datetime.combine(custom_from, time(0, 0, 0))
        intent.to_dt   = datetime.combine(custom_to,   time(23, 59, 59))
    elif time_override == "current":
        now = datetime.now()
        intent.time_range = "current"
        intent.from_dt = now - timedelta(minutes=30)
        intent.to_dt   = now
    else:
        intent.time_range = time_override
        intent.from_dt, intent.to_dt = resolve_time_range(time_override)

    # Respuestas conversacionales: streaming directo (se ven los tokens en tiempo real)
    if intent.action in ("chat", "unknown") and st.session_state.ollama_ok:
        with st.container():
            st.markdown(
                '<div class="ai-row"><div class="ai-avatar">AI</div></div>',
                unsafe_allow_html=True,
            )
            try:
                response_text = st.write_stream(
                    chat_as_agent_stream(query, machine_names=get_machine_names())
                )
            except Exception as e:
                response_text = f"Error al conectar con el modelo: {e}"
                st.error(response_text)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response_text,
            "result": {"type": "text"},
            "timestamp": datetime.now(),
        })
        st.rerun()
        return

    # Consultas de datos: spinner + ejecución regular
    with st.spinner("Procesando..."):
        result = _execute_intent(intent)
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result.get("content", ""),
        "result": result,
        "timestamp": datetime.now(),
    })
    st.rerun()


def _select_disambiguation(exact_name: str):
    intent = st.session_state.pending_intent
    if intent is None:
        return
    intent.machine_name = exact_name
    st.session_state.pending_intent = None
    st.session_state.chat_history.append({
        "role": "user",
        "content": f"Me refiero a **{exact_name}**",
        "timestamp": datetime.now(),
    })
    with st.spinner("⚙️ Trabajando en el análisis…"):
        result = _execute_intent(intent)
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result.get("content", ""),
        "result": result,
        "timestamp": datetime.now(),
    })


def _render_excel_download(result: dict):
    tag_data = result.get("tag_data") or {}
    all_data = result.get("all_data")
    if not tag_data and (all_data is None or all_data.empty):
        return
    machine_name = result.get("machine_name", "datos")
    safe_name = machine_name.replace(" ", "_").replace("/", "_")[:30]
    time_range = result.get("time_range", "data")

    def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if hasattr(df[col], "dt") and hasattr(df[col].dt, "tz"):
                if df[col].dt.tz is not None:
                    df[col] = df[col].dt.tz_localize(None)
        return df

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if tag_data:
            for tag_name, df in tag_data.items():
                short = tag_name.split(".")[-1] if "." in tag_name else tag_name
                sheet_name = short[:31].replace("/", "-").replace(":", "-")
                clean_df = _strip_tz(df)
                export_cols = [c for c in ["TimeStamp", "Value_Raw", "Value_Num", "Value_Str", "Tag_Type"] if c in clean_df.columns]
                clean_df[export_cols].to_excel(writer, index=False, sheet_name=sheet_name)
            if all_data is not None and not all_data.empty:
                _strip_tz(all_data).to_excel(writer, index=False, sheet_name="Resumen")
        else:
            _strip_tz(all_data).to_excel(writer, index=False, sheet_name="Datos")
    excel_bytes = output.getvalue()
    with st.container(border=True):
        st.markdown(f"**📥 Exportar datos** — {len(tag_data)} parámetros")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Descargar Excel",
                data=excel_bytes,
                file_name=f"{safe_name}_{time_range}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with c2:
            if all_data is not None and not all_data.empty:
                csv_bytes = _strip_tz(all_data).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Descargar CSV",
                    data=csv_bytes,
                    file_name=f"{safe_name}_{time_range}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


def _render_chat_history():
    for entry_idx, entry in enumerate(st.session_state.chat_history):
        if entry["role"] == "user":
            st.markdown(
                f'<div class="user-row"><div class="user-bubble">{entry["content"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            result = entry.get("result", {})
            res_type = result.get("type", "text")

            st.markdown(
                f'<div class="ai-row">'
                f'<div class="ai-avatar">AI</div>'
                f'<div class="ai-text">{entry["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if result.get("llm_summary") and result.get("show_analysis", True):
                with st.container(border=True):
                    st.markdown("##### 💡 Análisis inteligente")
                    st.markdown(result["llm_summary"])

            if res_type == "dashboard" and result.get("show_excel") and "all_data" in result:
                _render_excel_download(result)

            if res_type == "dashboard" and "tag_data" in result and result.get("show_dashboard", True):
                render_machine_dashboard(
                    result["machine_name"],
                    result["tag_data"],
                    time_range_label=result.get("time_range", "hoy"),
                    chart_instructions=result.get("chart_instructions", []),
                )
            elif res_type == "disambiguation" and "matches" in result:
                cols = st.columns(min(3, len(result["matches"])))
                for idx, (_, row) in enumerate(result["matches"].iterrows()):
                    with cols[idx % len(cols)]:
                        if st.button(
                            f"📍 {row['machine_name']}\n{row['line_name']}",
                            key=f"disamb_{entry_idx}_{idx}",
                            use_container_width=True,
                        ):
                            _select_disambiguation(row["machine_name"])
                            st.rerun()
            elif res_type == "machine_list":
                machines_df = discover_machines()
                chosen = render_machine_list(machines_df)
                if chosen:
                    _process_query(f"Dame los datos de hoy de {chosen}")
                    st.rerun()


# ================================================================
# Public render
# ================================================================
def render():
    ai_pill = '<span class="pill pill-ok">● IA Online</span>' if st.session_state.ollama_ok \
              else '<span class="pill pill-err">● IA Offline</span>'
    model_pill = f'<span class="pill pill-primary">{OLLAMA_MODEL}</span>'

    page_header(
        eyebrow="Mosaic AI · Agent",
        title="Agente conversacional",
        subtitle="Pregunta en lenguaje natural sobre cualquier máquina, parámetro o período. "
                 "El agente consulta PI, genera dashboards y analiza los datos por ti.",
        right_html=f"{ai_pill}&nbsp;{model_pill}",
    )

    # -------- Toolbar (tiempo + herramientas) --------
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 3])
        with c1:
            st.markdown("**⏱️ Período**")
            ranges = {
                "today": "Hoy",
                "yesterday": "Ayer",
                "last_hour": "Última hora",
                "last_24h": "Últimas 24h",
                "last_week": "Última semana",
                "current": "Ahora (30min)",
                "custom": "Personalizado",
            }
            st.session_state.ui_time_range = st.selectbox(
                "Rango",
                options=list(ranges.keys()),
                format_func=lambda k: ranges[k],
                index=list(ranges.keys()).index(st.session_state.ui_time_range),
                label_visibility="collapsed",
            )
            if st.session_state.ui_time_range == "custom":
                d1, d2 = st.columns(2)
                with d1:
                    st.session_state.ui_custom_from = st.date_input("Desde", st.session_state.ui_custom_from, label_visibility="collapsed")
                with d2:
                    st.session_state.ui_custom_to = st.date_input("Hasta", st.session_state.ui_custom_to, label_visibility="collapsed")

        with c2:
            st.markdown("**🧰 Herramientas activas**")
            st.session_state.opt_dashboard = st.toggle("Dashboard visual", st.session_state.opt_dashboard)
            st.session_state.opt_analysis  = st.toggle("Análisis IA",       st.session_state.opt_analysis)
            st.session_state.opt_excel     = st.toggle("Exportar Excel",    st.session_state.opt_excel)

        with c3:
            st.markdown("**🎯 Sugerencias**")
            sug_cols = st.columns(2)
            for i, sug in enumerate(QUERY_SUGGESTIONS[:6]):
                with sug_cols[i % 2]:
                    if st.button(sug, key=f"agent_sug_{i}", use_container_width=True):
                        _process_query(sug)
                        st.rerun()

    # -------- Historial --------
    if not st.session_state.chat_history:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.info("💬 Haz tu primera pregunta abajo o selecciona una sugerencia.")
    else:
        _render_chat_history()

    # -------- Procesar consulta pendiente (DESPUES del historial para que el usuario
    # vea su mensaje de inmediato antes de que aparezca la respuesta) --------
    if "_pending_query" in st.session_state:
        _process_pending()
        return  # _process_pending() llama a st.rerun() internamente

    # -------- Input --------
    prefill = st.session_state.pop("_prefill_query", "")
    prompt = st.chat_input("Pregúntame sobre cualquier máquina o parámetro…")
    if prefill and not prompt:
        prompt = prefill

    if prompt:
        _process_query(prompt)
        st.rerun()

    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑️ Limpiar conversación", key="agent_view_clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
