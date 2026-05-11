"""Agent — abre el agente externo en una nueva pestaña."""
import streamlit as st
from theme import page_header

AGENT_URL = "http://10.132.12.178:7860/"


def render():
    page_header(
        eyebrow="AI · Agent",
        title="Agent Kim",
        subtitle="Agente conversacional de inteligencia artificial para tu piso de planta.",
    )

    st.markdown(
        f"""
        <div style='
            display:flex;flex-direction:column;align-items:center;
            justify-content:center;padding:80px 20px;
            background:#f8fafc;border:2px dashed #cbd5e1;
            border-radius:16px;margin-top:10px;
        '>
            <div style='font-size:4rem;margin-bottom:16px;'>🤖</div>
            <div style='font-size:1.2rem;font-weight:700;color:#0f172a;margin-bottom:8px;'>
                Agent Kim
            </div>
            <div style='font-size:0.88rem;color:#64748b;text-align:center;
                        max-width:340px;margin-bottom:28px;'>
                El agente se ejecuta en un servicio externo.
                Haz clic en el botón para abrirlo en una nueva pestaña.
            </div>
            <a href="{AGENT_URL}" target="_blank" rel="noopener noreferrer"
               style='
                   display:inline-flex;align-items:center;gap:10px;
                   background:linear-gradient(135deg,#4f46e5,#7c3aed);
                   color:#fff;font-weight:700;font-size:1rem;
                   padding:14px 32px;border-radius:12px;
                   text-decoration:none;
                   box-shadow:0 4px 14px rgba(79,70,229,.35);
               '>
                🚀&nbsp; Abrir Agent Kim
            </a>
            <div style='margin-top:16px;font-size:0.75rem;color:#94a3b8;'>
                {AGENT_URL}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


_BASE_SYSTEM = """Eres el agente MIP (Machine Intelligence Platform) de Kimball Electronics Mexico.
Experto en manufactura SMT: Paste Printer, SPI, Pick & Place, Reflow, AOI, ICT.

Tienes acceso a datos en tiempo real de las maquinas de produccion de la planta via PI System.
Cuando el usuario pregunte por datos de una maquina y se te incluyan datos en el contexto, analizalos y responde con precision.
Cuando no haya datos de maquina en el contexto, responde con tu conocimiento de SMT.

REGLAS:
- Responde en el idioma del usuario (espanol por defecto).
- Se directo y conciso. Evita relleno.
- Usa formato markdown cuando ayude a la claridad.
- Si el usuario pide un analisis, usa los datos del contexto no inventes valores.
- Si no tienes datos suficientes para responder algo especifico, dilo claramente.
"""

# ─────────────────────────────────────────────────────────
# CSS — terminal + override de st.chat_input
# ─────────────────────────────────────────────────────────
_TERM_CSS = """
<style>
/* ── Keyframes ── */
@keyframes xterm-blink { 0%,100%{opacity:1} 50%{opacity:0} }
@keyframes xterm-ellipsis {
    0%   { content: ".";   }
    33%  { content: "..";  }
    66%  { content: "..."; }
    100% { content: "";    }
}
@keyframes xterm-spin {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* ── Shell container (history only, no bottom radius when chat_input is below) ── */
.xterm-shell {
    background: #0d1117;
    border: 1px solid #30363d;
    border-bottom: 1px solid #21262d;
    border-radius: 12px 12px 0 0;
    font-family: 'Consolas','JetBrains Mono','Courier New',monospace;
    font-size: 0.855rem;
    line-height: 1.7;
    color: #c9d1d9;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.45);
}

/* ── Title bar ── */
.xterm-titlebar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    background: #161b22;
    border-bottom: 1px solid #21262d;
    user-select: none;
}
.xterm-btn { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
.xterm-btn-r { background: #f85149; }
.xterm-btn-y { background: #d29922; }
.xterm-btn-g { background: #3fb950; }
.xterm-title {
    flex: 1; text-align: center;
    font-size: 0.74rem; color: #8b949e;
    letter-spacing: 0.04em;
}
.xterm-status-pill {
    display: flex; align-items: center; gap: 5px;
    font-size: 0.71rem; color: #8b949e;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 999px;
    padding: 2px 9px;
}
.xterm-status-dot { width: 7px; height: 7px; border-radius: 50%; }
.xterm-status-dot-anim {
    animation: xterm-blink 1s ease-in-out infinite;
}

/* ── Body ── */
.xterm-body {
    padding: 14px 18px 10px;
    min-height: 340px;
    max-height: 56vh;
    overflow-y: auto;
}

/* ── Messages ── */
.xterm-line-user { color: #58a6ff; margin: 10px 0 1px; }
.xterm-prompt-glyph { color: #3fb950; }
.xterm-line-ts { color: #484f58; font-size: 0.69rem; margin: 0 0 4px 0; }
.xterm-line-ctx {
    color: #8b949e; font-size: 0.75rem;
    background: #161b22; border: 1px solid #21262d;
    border-radius: 5px; padding: 3px 10px;
    margin: 2px 0 6px 14px; display: inline-block;
}
.xterm-line-ai {
    color: #c9d1d9;
    white-space: pre-wrap;
    word-break: break-word;
    margin: 2px 0 12px 0;
    padding-left: 14px;
    border-left: 2px solid #21262d;
}
.xterm-cursor { animation: xterm-blink 0.9s step-start infinite; color: #3fb950; }
.xterm-thinking::after {
    content: ".";
    animation: xterm-ellipsis 1.2s steps(1) infinite;
    color: #58a6ff;
}
.xterm-thinking-wrap {
    color: #58a6ff;
    padding-left: 14px;
    font-style: italic;
    margin-bottom: 10px;
}
.xterm-fetch-wrap {
    color: #f0883e;
    padding-left: 14px;
    font-style: italic;
    margin-bottom: 10px;
}
.xterm-err-wrap { color: #f85149; padding-left: 14px; margin-bottom: 10px; }

/* ── Input bar strip (sits between shell and the Streamlit chat_input) ── */
.xterm-input-strip {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #0d1117;
    border-left: 1px solid #30363d;
    border-right: 1px solid #30363d;
    padding: 6px 14px 4px;
    font-family: 'Consolas','JetBrains Mono','Courier New',monospace;
    font-size: 0.88rem;
}
.xterm-input-prompt { color: #3fb950; white-space: nowrap; }
.xterm-input-path   { color: #58a6ff; }

/* ── Override st.chat_input to continue the terminal box ── */
[data-testid="stChatInputContainer"] {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 4px 10px 8px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.45) !important;
    margin-top: 0 !important;
}
[data-testid="stChatInputContainer"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInputContainer"] textarea {
    background: transparent !important;
    color: #e6edf3 !important;
    font-family: 'Consolas','JetBrains Mono','Courier New',monospace !important;
    font-size: 0.88rem !important;
    caret-color: #3fb950 !important;
    border: none !important;
    padding-left: 0 !important;
}
[data-testid="stChatInputContainer"] textarea::placeholder {
    color: #484f58 !important;
    font-family: 'Consolas','JetBrains Mono','Courier New',monospace !important;
}
[data-testid="stChatInputContainer"] button {
    background: transparent !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #3fb950 !important;
}
[data-testid="stChatInputContainer"] button:hover {
    background: #21262d !important;
    border-color: #3fb950 !important;
}

/* ── Bottom bar (clear + model info) ── */
.xterm-bottom-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 6px;
    padding: 0 2px;
    font-family: 'Consolas','JetBrains Mono','Courier New',monospace;
    font-size: 0.72rem;
    color: #484f58;
}
</style>
"""

# ─────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────
def _esc(text: str) -> str:
    return _html_module.escape(str(text)).replace("\n", "<br>")


def _build_history_html(messages: list, ai_state: str, streaming_chunk: str = "") -> str:
    meta = _STATE_META.get(ai_state, _STATE_META[STATE_IDLE])
    dot_color  = meta["dot"]
    state_label = meta["label"]
    dot_class  = "xterm-status-dot xterm-status-dot-anim" if meta["anim"] else "xterm-status-dot"

    # History lines
    history_html = ""
    for msg in messages:
        if msg["role"] == "user":
            history_html += (
                f"<div class='xterm-line-user'>"
                f"<span class='xterm-prompt-glyph'>mip@kimball:~$</span>&nbsp;{_esc(msg['content'])}"
                f"</div>"
                f"<div class='xterm-line-ts'>{msg.get('ts','')}</div>"
            )
        else:
            ctx = msg.get("ctx", "")
            ctx_html = (
                f"<div class='xterm-line-ctx'>&#128225;&nbsp;datos cargados: {_esc(ctx)}</div>"
                if ctx else ""
            )
            history_html += (
                f"{ctx_html}"
                f"<div class='xterm-line-ai'>{_esc(msg['content'])}</div>"
            )

    # Live indicator
    live_html = ""
    if ai_state == STATE_STREAMING and streaming_chunk:
        live_html = (
            f"<div class='xterm-line-ai'>{_esc(streaming_chunk)}"
            f"<span class='xterm-cursor'>&#9646;</span></div>"
        )
    elif ai_state == STATE_THINKING:
        live_html = (
            "<div class='xterm-thinking-wrap'>"
            "<span class='xterm-thinking'>deepseek thinking</span></div>"
        )
    elif ai_state == STATE_FETCHING:
        live_html = (
            "<div class='xterm-fetch-wrap'>"
            "<span class='xterm-thinking'>fetching PI data</span></div>"
        )
    elif ai_state == STATE_ERROR:
        live_html = "<div class='xterm-err-wrap'>&#9888; Error en la ultima consulta. Intenta de nuevo.</div>"

    return f"""
{_TERM_CSS}
<div class="xterm-shell">
  <div class="xterm-titlebar">
    <span class="xterm-btn xterm-btn-r"></span>
    <span class="xterm-btn xterm-btn-y"></span>
    <span class="xterm-btn xterm-btn-g"></span>
    <div class="xterm-title">MIP Terminal &mdash; {OLLAMA_MODEL}</div>
    <div class="xterm-status-pill">
      <span class="{dot_class}" style="background:{dot_color};box-shadow:0 0 5px {dot_color};"></span>
      {state_label}
    </div>
  </div>
  <div class="xterm-body" id="xterm-body">
    <div style="color:#3fb950;margin-bottom:8px;font-size:0.77rem;">
      Machine Intelligence Platform &mdash; {OLLAMA_MODEL}<br>
      <span style="color:#484f58;">Connected to PI System. Type a question below.</span>
    </div>
    <hr style="border:none;border-top:1px solid #21262d;margin:6px 0 12px;">
    {history_html}
    {live_html}
  </div>
</div>
<div class="xterm-input-strip">
  <span class="xterm-input-prompt">mip</span>
  <span style="color:#484f58;">@</span>
  <span class="xterm-input-path">kimball</span>
  <span style="color:#484f58;">:~$</span>
</div>
"""

# ─────────────────────────────────────────────────────────
# Machine context helpers
# ─────────────────────────────────────────────────────────
def _machine_context_text(machine_name: str, machine_path: str, hours: float = 1.0) -> str:
    try:
        tags_df = get_machine_tags(machine_path)
        if tags_df.empty:
            return f"[Sin tags para {machine_name}]"
        tag_names = tags_df["piPoint"].astype(str).tolist()
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(hours=hours)
        all_data = fetch_tag_values(tag_names, from_dt, to_dt)
        if all_data.empty:
            return f"[Sin datos en la ultima {hours:.0f}h para {machine_name}]"
        lines = [f"=== Datos de {machine_name} (ultima {hours:.0f}h) ==="]
        for tn in tag_names:
            sub = all_data[all_data["Tag_Name"] == tn].copy()
            if sub.empty:
                continue
            short = tn.split(".")[-1] if "." in tn else tn
            mode = infer_tag_mode(sub)
            if mode == "numeric":
                s = numeric_summary(sub)
                if s:
                    lines.append(
                        f"  {short}: ultimo={s.get('last',0):.3f}  "
                        f"min={s.get('min',0):.3f}  max={s.get('max',0):.3f}  "
                        f"prom={s.get('mean',0):.3f}  CV%={s.get('cv_pct',0):.1f}"
                    )
            else:
                s = categorical_summary(sub)
                if s:
                    lines.append(f"  {short}: ultimo={s.get('last','?')}  estados={s.get('unique_count','?')}")
        return "\n".join(lines)
    except Exception as e:
        return f"[Error: {e}]"


def _detect_machine(query: str):
    try:
        matches = smart_find_machine(query)
        if not matches.empty:
            row = matches.iloc[0]
            return row["machine_name"], row["machine_path"]
    except Exception:
        pass
    return None, None


# ─────────────────────────────────────────────────────────
# Estado de sesion
# ─────────────────────────────────────────────────────────
def _init():
    defaults = {
        "term_messages":  [],
        "term_state":     STATE_IDLE,
        "term_streaming": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────
# Procesar mensaje
# ─────────────────────────────────────────────────────────
def _handle_message(user_input: str, terminal_placeholder):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.term_messages.append({"role": "user", "content": user_input, "ts": ts})

    # Estado: fetching
    st.session_state.term_state = STATE_FETCHING
    terminal_placeholder.markdown(
        _build_history_html(st.session_state.term_messages, STATE_FETCHING),
        unsafe_allow_html=True,
    )

    # Contexto de maquina
    machine_name, machine_path = _detect_machine(user_input)
    ctx_label = ""
    extra_context = ""
    if machine_name and machine_path:
        ctx_label = machine_name
        extra_context = "\n\n" + _machine_context_text(machine_name, machine_path)

    names = get_machine_names()
    machines_hint = f"\n\nMaquinas disponibles: {', '.join(names[:40])}" if names else ""
    system = _BASE_SYSTEM + machines_hint + extra_context

    MAX_TURNS = 14
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.term_messages[-MAX_TURNS:]
    ]

    # Estado: thinking
    st.session_state.term_state = STATE_THINKING
    terminal_placeholder.markdown(
        _build_history_html(st.session_state.term_messages, STATE_THINKING),
        unsafe_allow_html=True,
    )

    # Streaming
    st.session_state.term_state = STATE_STREAMING
    full_response = ""
    error_msg = ""
    try:
        stream = terminal_chat_stream(history, system)
        for chunk in stream:
            full_response += chunk
            terminal_placeholder.markdown(
                _build_history_html(st.session_state.term_messages, STATE_STREAMING, full_response),
                unsafe_allow_html=True,
            )
    except Exception as e:
        error_msg = str(e)

    # Guardar respuesta
    final_content = full_response if full_response else f"[Error: {error_msg}]"
    st.session_state.term_messages.append({
        "role": "assistant",
        "content": final_content,
        "ts": datetime.now().strftime("%H:%M:%S"),
        "ctx": ctx_label,
    })
    st.session_state.term_state = STATE_ERROR if (not full_response and error_msg) else STATE_IDLE
    st.session_state.term_streaming = ""
    st.rerun()


# ─────────────────────────────────────────────────────────
# Render principal
# ─────────────────────────────────────────────────────────
def render():
    _init()

    state = st.session_state.term_state
    is_busy = state not in (STATE_IDLE, STATE_ERROR)

    # Header
    st.markdown(
        f"<h2 style='margin-bottom:0.1rem;font-family:Consolas,monospace;'>&#11035; MIP Terminal</h2>"
        f"<p style='color:#6e7681;font-size:0.82rem;margin-top:0;'>"
        f"deepseek-coder-v2 via Ollama &middot; PI System context &middot; multi-turn</p>",
        unsafe_allow_html=True,
    )

    # Terminal placeholder (history + titlebar)
    terminal_placeholder = st.empty()
    terminal_placeholder.markdown(
        _build_history_html(
            st.session_state.term_messages,
            state,
            st.session_state.term_streaming,
        ),
        unsafe_allow_html=True,
    )

    # Input — nativo de Streamlit, estilizado via CSS para continuar el terminal
    placeholder_text = (
        "mip@kimball:~$  escribe un mensaje..."
        if not is_busy
        else f"  {state}..."
    )
    prompt = st.chat_input(placeholder_text, key="term_input", disabled=is_busy)

    # Bottom bar
    col_info, col_clear = st.columns([5, 1])
    with col_info:
        n_msg = len([m for m in st.session_state.term_messages if m["role"] == "user"])
        st.markdown(
            f"<div class='xterm-bottom-bar'>"
            f"{OLLAMA_MODEL} &nbsp;|&nbsp; {n_msg} mensajes &nbsp;|&nbsp; "
            f"PI System {'&#x2022; online' if st.session_state.get('machines_loaded') else '&#x2022; offline'}"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col_clear:
        if st.button("🗑 Clear", use_container_width=True, help="Limpiar conversacion"):
            st.session_state.term_messages = []
            st.session_state.term_state = STATE_IDLE
            st.rerun()

    if prompt and prompt.strip() and not is_busy:
        if not st.session_state.get("ollama_ok"):
            st.error("Ollama no disponible. Verifica que el servidor este corriendo en localhost:11434")
            return
        _handle_message(prompt.strip(), terminal_placeholder)
