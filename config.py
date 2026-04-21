"""
Configuración central de la aplicación.
"""
from datetime import datetime, timedelta, time

# =========================================================
# PI WEB API
# =========================================================
BASE_URL = "https://kp.kemx.keint.com/PI_WebApi/api/Generic"
ENDPOINT_CHILDREN = f"{BASE_URL}/Fetch_Child_Elements"
ENDPOINT_ATTRIBUTES = f"{BASE_URL}/Fetch_Element_Attributes"
ENDPOINT_TAG_VALUES = f"{BASE_URL}/Fetch_Generic_Tag_Class_with_PluginName"

PLUGIN_NAME = "MBBP Data Fetch"
ROOT_PATH = r"\\NTS5120\Kimball BD Produccion\Kimball Electronics Mexico\Production\SMT\Plant 2"

VERIFY_SSL = False
API_TIMEOUT = 30

# Nota de zona horaria: PI envía timestamps con el offset local embebido
# (ej: "2026-04-15T11:00:00-05:00"). El parser los preserva tal cual.
# No se requiere conversión manual de timezone.

# =========================================================
# OLLAMA / LLM
# =========================================================
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "deepseek-coder-v2"
LLM_TEMPERATURE = 0.1
LLM_TIMEOUT = 180

# =========================================================
# DEFAULTS DE TIEMPO
# =========================================================
DEFAULT_DAYS_BACK = 1
DEFAULT_FROM_TIME = time(0, 0, 0)
DEFAULT_TO_TIME = time(23, 59, 59)

def get_today_range():
    now = datetime.now()
    return (
        now.replace(hour=0, minute=0, second=0, microsecond=0),
        now
    )

def get_default_range():
    now = datetime.now()
    return (
        (now - timedelta(days=DEFAULT_DAYS_BACK)).replace(hour=0, minute=0, second=0, microsecond=0),
        now
    )

# =========================================================
# SUGERENCIAS PREDEFINIDAS
# =========================================================
QUERY_SUGGESTIONS = [
    "¿Cuáles son los datos de hoy de la Paste Printer?",
    "Muéstrame el estado actual de todas las máquinas",
    "¿Cuál es la temperatura de la Reflow Oven hoy?",
    "Dame un resumen de la Pick and Place",
    "¿Qué máquinas están activas ahora?",
    "Muéstrame los datos de la última hora del SPI",
    "¿Hay algún valor fuera de rango en la Paste Printer?",
    "Compara los datos de hoy vs ayer de la Wave Solder",
]
