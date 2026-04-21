"""
Interpreta preguntas en lenguaje natural y las convierte
en intenciones estructuradas que el sistema puede ejecutar.
"""
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict

from llm_engine import chat_completion_json
from machine_registry import build_machine_context_for_llm


# ----------------------------------------------------------
# Intención estructurada
# ----------------------------------------------------------
@dataclass
class QueryIntent:
    action: str = "show_dashboard"  # show_dashboard | list_machines | show_attribute | compare | unknown
    machine_name: str | None = None
    line_hint: str | None = None    # Pista de línea para desambiguar (ej: "L1L", "Linea 1 Left")
    attribute_name: str | None = None
    time_range: str = "today"       # today | yesterday | last_hour | last_week | custom
    from_dt: datetime | None = None
    to_dt: datetime | None = None
    summary_text: str = ""          # Texto amigable describiendo lo que se va a hacer
    chart_instructions: list = field(default_factory=list)  # Instrucciones de visualización del LLM
    raw_query: str = ""
    error: str = ""
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------
# Resolver rango de tiempo
# ----------------------------------------------------------
def resolve_time_range(time_range: str) -> tuple[datetime, datetime]:
    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    mapping = {
        "today":      (today_start, now),
        "yesterday":  (today_start - timedelta(days=1), today_start),
        "last_hour":  (now - timedelta(hours=1), now),
        "last_week":  (today_start - timedelta(days=7), now),
        "last_24h":   (now - timedelta(hours=24), now),
        "last_shift": (now - timedelta(hours=12), now),
    }
    return mapping.get(time_range, (today_start, now))


# ----------------------------------------------------------
# System prompt para el parser
# ----------------------------------------------------------
PARSER_SYSTEM_PROMPT = """Eres un parser de intenciones para un sistema de monitoreo de máquinas SMT de manufactura electrónica.

Tu trabajo es recibir una pregunta en español (o inglés) y devolver un JSON con la intención del usuario.

{machine_context}

REGLAS:
1. El campo "action" puede ser: "show_dashboard", "list_machines", "show_attribute", "compare", "status_check", "chat", "unknown"
   - Usa "chat" cuando el usuario haga una pregunta general, saludo, consulta conceptual sobre manufactura SMT,
     o cualquier conversación que NO requiera datos de una máquina concreta (ej: "Hola", "¿Qué puedes hacer?",
     "¿Qué es el reflow?", "Dame consejos sobre pasta de soldadura").
   - Solo usa "unknown" si la pregunta es completamente incomprensible o sin sentido.
2. El campo "machine_name" puede ser:
   - El nombre EXACTO de una máquina del catálogo (preferido)
   - O una PALABRA CLAVE / nombre parcial que el usuario use (ej: "SPI", "Reflow", "Printer", "AOI")
   - NUNCA dejes machine_name como null si el usuario menciona algún tipo de máquina.
3. El campo "line_hint" es MUY IMPORTANTE: si el usuario menciona una línea específica, pon el identificador aquí.
   - Ejemplos: "Linea 1 Left" -> "L1L", "Linea 1 Right" -> "L1R", "Linea 7 Left" -> "L7L", "Linea 3" -> "L3"
   - También acepta: "linea 1", "l1l", "line 1 left", "1L", etc.
   - Si NO menciona línea, pon null.
4. Si el usuario pregunta por un atributo específico (temperatura, velocidad, etc.), ponlo en "attribute_name"
5. El campo "time_range" puede ser: "today", "yesterday", "last_hour", "last_week", "last_24h", "last_shift"
6. El campo "summary_text" es un texto corto y amigable en español explicando qué vas a hacer
7. El campo "chart_instructions" es una LISTA de instrucciones de visualización. Cada elemento es un objeto con:
   - "type": tipo de gráfica ("line", "outliers", "histogram", "comparison", "bar")
   - "description": qué debe mostrar la gráfica
   Si el usuario pide algo visual específico (outliers, límites, desviación estándar, histograma, etc.) incluyélo aquí.
   Si NO pide nada visual específico, deja la lista vacía [].
8. Si no entiendes la pregunta, usa action="unknown"

RESPONDE SOLO con un JSON válido, sin texto adicional.

Ejemplo 1 - Consulta simple:
{{
    "action": "show_dashboard",
    "machine_name": "Paste Printer",
    "line_hint": null,
    "attribute_name": null,
    "time_range": "today",
    "summary_text": "Mostrando los datos de hoy de la Paste Printer",
    "chart_instructions": []
}}

Ejemplo 2 - Con línea específica y gráfica de outliers:
{{
    "action": "show_dashboard",
    "machine_name": "Paste Printer",
    "line_hint": "L1L",
    "attribute_name": null,
    "time_range": "today",
    "summary_text": "Analizando outliers de la Paste Printer en la Línea 1 Left",
    "chart_instructions": [
        {{"type": "outliers", "description": "Gráfica con límites upper/lower basados en desviación estándar"}}
    ]
}}

Ejemplo 3 - Comparación:
{{
    "action": "compare",
    "machine_name": "Paste Printer",
    "line_hint": "L7R",
    "attribute_name": null,
    "time_range": "today",
    "summary_text": "Comparando datos de ayer vs hoy de la Paste Printer L7R",
    "chart_instructions": [
        {{"type": "comparison", "description": "Comparar valores de ayer vs hoy"}}
    ]
}}"""


def parse_query(user_query: str) -> QueryIntent:
    """
    Usa el LLM para interpretar una pregunta en lenguaje natural.
    """
    machine_context = build_machine_context_for_llm()
    system = PARSER_SYSTEM_PROMPT.format(machine_context=machine_context)

    result = chat_completion_json(system, user_query)

    intent = QueryIntent(raw_query=user_query)

    if "error" in result and "raw" in result:
        intent.error = f"No pude interpretar tu pregunta. Intenta reformularla."
        intent.action = "unknown"
        return intent

    intent.action = result.get("action", "unknown")
    intent.machine_name = result.get("machine_name")
    intent.line_hint = result.get("line_hint")
    intent.attribute_name = result.get("attribute_name")
    intent.time_range = result.get("time_range", "today")
    intent.summary_text = result.get("summary_text", "")
    intent.chart_instructions = result.get("chart_instructions", [])

    # Resolver fechas
    from_dt, to_dt = resolve_time_range(intent.time_range)
    intent.from_dt = from_dt
    intent.to_dt = to_dt

    return intent


def parse_query_fallback(user_query: str, machine_names: list[str]) -> QueryIntent:
    """
    Parser sin LLM: busca nombres de máquina en el texto.
    Útil como fallback si Ollama no está disponible.
    """
    import re
    query_lower = user_query.lower()
    intent = QueryIntent(raw_query=user_query)

    # Extraer line_hint
    line_patterns = [
        (r"l(\d+)(l|r)\b", lambda m: f"L{m.group(1)}{m.group(2).upper()}"),  # L1L, L7R
        (r"linea\s+(\d+)\s+(left|right|izquierda|derecha)", lambda m: f"L{m.group(1)}{'L' if m.group(2) in ('left','izquierda') else 'R'}"),
        (r"line\s+(\d+)\s+(left|right)", lambda m: f"L{m.group(1)}{'L' if m.group(2)=='left' else 'R'}"),
        (r"línea\s+(\d+)\s+(left|right|izquierda|derecha)", lambda m: f"L{m.group(1)}{'L' if m.group(2) in ('left','izquierda') else 'R'}"),
        (r"linea\s+(\d+)", lambda m: f"L{m.group(1)}"),
        (r"línea\s+(\d+)", lambda m: f"L{m.group(1)}"),
    ]
    for pattern, transform in line_patterns:
        match = re.search(pattern, query_lower)
        if match:
            intent.line_hint = transform(match)
            break

    # Extraer chart_instructions del texto
    if any(w in query_lower for w in ["outlier", "desviación", "desviacion", "std", "sigma", "límite", "limite", "upper", "lower"]):
        intent.chart_instructions.append({"type": "outliers", "description": "Gráfica con límites basados en desviación estándar"})
    if any(w in query_lower for w in ["histograma", "histogram", "distribución", "distribucion", "frecuencia"]):
        intent.chart_instructions.append({"type": "histogram", "description": "Histograma de distribución de valores"})

    # Detectar máquina — primero exacto, luego palabra clave
    for name in machine_names:
        if name.lower() in query_lower:
            intent.machine_name = name
            break

    # Si no hubo match exacto, buscar palabras clave comunes de máquinas
    if not intent.machine_name:
        machine_keywords = ["spi", "aoi", "reflow", "printer", "paste", "pick", "place",
                           "wave", "solder", "conveyor", "loader", "unloader", "oven",
                           "dispenser", "inspection", "buffer", "magazine"]
        for kw in machine_keywords:
            if kw in query_lower:
                intent.machine_name = kw
                break

    # Detectar rango de tiempo
    time_keywords = {
        "hoy": "today",
        "today": "today",
        "ayer": "yesterday",
        "yesterday": "yesterday",
        "última hora": "last_hour",
        "last hour": "last_hour",
        "semana": "last_week",
        "week": "last_week",
    }
    for keyword, range_val in time_keywords.items():
        if keyword in query_lower:
            intent.time_range = range_val
            break

    # Detectar acción
    if any(w in query_lower for w in ["lista", "máquinas", "machines", "cuáles hay", "disponibles"]):
        intent.action = "list_machines"
        intent.summary_text = "Mostrando las máquinas disponibles"
    elif intent.machine_name:
        intent.action = "show_dashboard"
        intent.summary_text = f"Mostrando datos de {intent.machine_name}"
    else:
        intent.action = "unknown"
        intent.summary_text = "No pude identificar una máquina en tu pregunta"

    from_dt, to_dt = resolve_time_range(intent.time_range)
    intent.from_dt = from_dt
    intent.to_dt = to_dt

    return intent
