"""
Motor de LLM: interfaz con Ollama (deepseek-coder-v2).
"""
import json
import requests
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT


def chat_completion(
    system_prompt: str,
    user_message: str,
    temperature: float | None = None,
    force_json: bool = False,
) -> str:
    """
    Envía un mensaje al modelo Ollama y devuelve la respuesta como texto.
    Si force_json=True, usa el parametro format=json de Ollama para forzar salida JSON valida.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {
            "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
        },
    }
    if force_json:
        payload["format"] = "json"

    resp = requests.post(url, json=payload, timeout=LLM_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "").strip()


def _extract_json(text: str) -> dict | None:
    """
    Intenta extraer un objeto JSON valido de un texto 'sucio'.
    Maneja: code fences, comentarios, texto alrededor, trailing commas,
    comillas simples, JSON partido por saltos de linea.
    """
    import re
    if not text:
        return None
    t = text.strip()

    # 1) Quitar code fences ```json ... ```
    if t.startswith("```"):
        lines = t.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()

    # 2) Intento directo
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass

    # 3) Buscar el mayor bloque {...} balanceado
    candidates = []
    stack = []
    start_idx = None
    for i, ch in enumerate(t):
        if ch == "{":
            if not stack:
                start_idx = i
            stack.append("{")
        elif ch == "}" and stack:
            stack.pop()
            if not stack and start_idx is not None:
                candidates.append(t[start_idx:i + 1])
                start_idx = None

    # ordenar por tamano descendente (el mas completo suele ser el ultimo/mayor)
    for cand in sorted(candidates, key=len, reverse=True):
        # Limpiezas tipicas
        cleaned = cand
        # quitar comentarios // y /* */
        cleaned = re.sub(r"//[^\n]*", "", cleaned)
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
        # quitar trailing commas antes de } o ]
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # intentar con comillas simples -> dobles
            try:
                return json.loads(cleaned.replace("'", '"'))
            except json.JSONDecodeError:
                continue

    return None


def chat_completion_json(
    system_prompt: str,
    user_message: str,
    temperature: float | None = None,
) -> dict:
    """
    Igual que chat_completion pero parsea la respuesta como JSON.
    El system_prompt debe instruir al modelo a devolver JSON valido.
    Hace 1 retry con feedback si el primer intento falla.
    """
    raw = chat_completion(system_prompt, user_message, temperature, force_json=True)
    parsed = _extract_json(raw)
    if parsed is not None:
        return parsed

    # Retry: pedirle al modelo que corrija su propia respuesta
    retry_user = (
        f"Tu respuesta anterior no fue JSON valido. Esta fue:\n\n{raw}\n\n"
        f"Por favor, devuelve UNICAMENTE un objeto JSON valido, sin texto extra, "
        f"sin markdown, sin comentarios. Responde al siguiente prompt original:\n\n{user_message}"
    )
    try:
        raw2 = chat_completion(system_prompt, retry_user, temperature=0.0, force_json=True)
        parsed2 = _extract_json(raw2)
        if parsed2 is not None:
            return parsed2
        return {"error": "No se pudo parsear la respuesta del modelo", "raw": raw2}
    except Exception as e:
        return {"error": f"No se pudo parsear la respuesta del modelo: {e}", "raw": raw}


def chat_completion_stream(
    system_prompt: str,
    user_message: str,
    temperature: float | None = None,
):
    """
    Envia un mensaje al modelo Ollama en modo streaming.
    Genera (yield) tokens de texto uno a uno para mostrarlos en tiempo real.
    Compatible con st.write_stream() de Streamlit.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": True,
        "options": {
            "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
        },
    }
    try:
        with requests.post(url, json=payload, timeout=LLM_TIMEOUT, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break
    except Exception as exc:
        yield f"\n\n*(Error de conexion con el modelo: {exc})*"


def generate_dashboard_summary(
    machine_name: str,
    data_summary: str,
) -> str:
    """
    Pide al LLM que genere un resumen en lenguaje natural
    de los datos de una máquina.
    """
    system = (
        "Eres un asistente experto en manufactura electrónica SMT. "
        "Tu trabajo es explicar datos de máquinas de producción de forma clara y sencilla, "
        "en español. Usa un tono amigable y profesional. "
        "Si ves valores fuera de rango o patrones interesantes, menciónalos. "
        "Sé conciso pero informativo (máximo 4-5 oraciones)."
    )
    user = (
        f"Genera un resumen breve de los datos de la máquina '{machine_name}'.\n\n"
        f"Datos:\n{data_summary}"
    )
    return chat_completion(system, user, temperature=0.3)


def answer_user_question(
    user_question: str,
    machine_name: str,
    data_summary: str,
) -> str:
    """
    El LLM responde directamente la pregunta del usuario usando los datos reales.
    Tiene libertad total para interpretar, comparar, opinar y recomendar.
    """
    system = (
        "Eres un asistente experto en manufactura electrónica SMT (Surface Mount Technology). "
        "Trabajas en una planta de Kimball Electronics en México.\n\n"
        "REGLAS:\n"
        "1. RESPONDE DIRECTAMENTE la pregunta del usuario. No hagas un resumen genérico.\n"
        "2. Usa los datos proporcionados para dar una respuesta ESPECÍFICA y ÚTIL.\n"
        "3. Si el usuario pregunta por diferencias, COMPARA valores concretos.\n"
        "4. Si el usuario pregunta por anomalías, SEÑALA valores fuera de lo normal.\n"
        "5. Si el usuario pregunta por tendencias, DESCRIBE la dirección de los datos.\n"
        "6. Usa un tono profesional pero amigable, en español.\n"
        "7. Puedes usar listas, negritas y formato para que sea fácil de leer.\n"
        "8. Si los datos no son suficientes para responder, dilo honestamente.\n"
        "9. Da recomendaciones cuando sea apropiado.\n"
        "10. NO repitas los datos crudos, INTERPRÉTALOS."
    )
    user = (
        f"PREGUNTA DEL USUARIO: {user_question}\n\n"
        f"MÁQUINA: {machine_name}\n\n"
        f"DATOS DISPONIBLES:\n{data_summary}\n\n"
        f"Responde la pregunta del usuario de manera directa y específica."
    )
    return chat_completion(system, user, temperature=0.4)


def llm_pick_machine(user_query: str, candidates: list[str]) -> str | None:
    """
    Dado el query del usuario y una lista de maquinas candidatas,
    pide al LLM que elija la mas probable o indique si ninguna aplica.
    Retorna el nombre exacto elegido, o None si el LLM no puede decidir.
    """
    if not candidates:
        return None
    options_text = "\n".join(f"- {c}" for c in candidates)
    system = (
        "Eres un asistente experto en manufactura SMT. "
        "El usuario pidio informacion de una maquina. "
        "Dado el listado de maquinas disponibles, elige la que MAS probablemente quiso decir. "
        "Responde UNICAMENTE con el nombre exacto de la maquina del listado, sin agregar nada mas. "
        "Si ninguna aplica, responde exactamente: NINGUNA"
    )
    user = (
        f"PREGUNTA DEL USUARIO: {user_query}\n\n"
        f"MAQUINAS DISPONIBLES:\n{options_text}\n\n"
        "Cual maquina quiso decir el usuario? Responde solo el nombre exacto."
    )
    try:
        raw = chat_completion(system, user, temperature=0.0).strip()
        # Verificar que sea uno de los candidatos (busqueda case-insensitive)
        raw_lower = raw.lower()
        for c in candidates:
            if c.lower() == raw_lower or raw_lower in c.lower():
                return c
        if "ninguna" in raw_lower:
            return None
        # Si el modelo devolvio algo que no matchea exactamente, buscar por contenido
        for c in candidates:
            if any(word in c.lower() for word in raw_lower.split() if len(word) > 2):
                return c
    except Exception:
        pass
    return None


def chat_as_agent(
    user_message: str,
    machine_names: list | None = None,
) -> str:
    """
    Responde como el agente de Machine Intelligence de forma libre:
    saludos, preguntas conceptuales sobre SMT, ayuda sobre el sistema, etc.
    """
    machines_hint = ""
    if machine_names:
        machines_hint = (
            f"\n\nMáquinas disponibles en planta: {', '.join(machine_names[:30])}"
        )
    system = (
        "Eres el Agente de Machine Intelligence de Kimball Electronics México, "
        "experto en manufactura electrónica SMT (Surface Mount Technology) y en el análisis "
        "de datos de producción de líneas SMT.\n\n"
        "Puedes:\n"
        "- Responder saludos y preguntas generales de forma natural y amigable.\n"
        "- Explicar conceptos de manufactura SMT: Paste Printer, Pick & Place, Reflow, SPI, AOI, etc.\n"
        "- Dar consejos sobre procesos, parámetros, defectos comunes y mejores prácticas.\n"
        "- Orientar al usuario sobre qué consultas puede hacer en este sistema.\n"
        "- Responder en el idioma en que el usuario te hable (español o inglés).\n\n"
        "Recuerda que en este sistema el usuario puede preguntarte por datos en tiempo real "
        "de cualquier máquina, parámetros históricos, outliers, correlaciones y más.\n"
        "Cuando sea relevante, guía al usuario hacia esas funciones."
        f"{machines_hint}"
    )
    return chat_completion(system, user_message, temperature=0.5)


def chat_as_agent_stream(
    user_message: str,
    machine_names: list | None = None,
):
    """
    Version streaming de chat_as_agent.
    Retorna un generador de tokens compatible con st.write_stream().
    """
    machines_hint = ""
    if machine_names:
        machines_hint = (
            f"\n\nMaquinas disponibles en planta: {', '.join(machine_names[:30])}"
        )
    system = (
        "Eres el Agente de Machine Intelligence de Kimball Electronics Mexico, "
        "experto en manufactura electronica SMT (Surface Mount Technology) y en el analisis "
        "de datos de produccion de lineas SMT.\n\n"
        "Puedes:\n"
        "- Responder saludos y preguntas generales de forma natural y amigable.\n"
        "- Explicar conceptos de manufactura SMT: Paste Printer, Pick & Place, Reflow, SPI, AOI, etc.\n"
        "- Dar consejos sobre procesos, parametros, defectos comunes y mejores practicas.\n"
        "- Orientar al usuario sobre que consultas puede hacer en este sistema.\n"
        "- Responder en el idioma en que el usuario te hable (espanol o ingles).\n\n"
        "Recuerda que en este sistema el usuario puede preguntarte por datos en tiempo real "
        "de cualquier maquina, parametros historicos, outliers, correlaciones y mas.\n"
        "Cuando sea relevante, guia al usuario hacia esas funciones."
        f"{machines_hint}"
    )
    return chat_completion_stream(system, user_message, temperature=0.5)


def is_ollama_available() -> bool:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False
