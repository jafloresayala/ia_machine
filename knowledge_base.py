"""Base de conocimiento de parámetros típicos de máquinas SMT.

Es un catálogo curado de parámetros comunes que aparecen en líneas SMT
(Paste Printer, Pick & Place, Reflow, SPI, AOI, etc.) usado por
la vista "Knowledge" y por el agente para explicar parámetros al usuario.
"""

PARAMETER_KNOWLEDGE = {
    # ========== PASTE PRINTER ==========
    "paste_printer": {
        "machine_keywords": ["paste printer", "printer", "stencil", "dek", "mpm"],
        "description": (
            "La impresora de pasta deposita pasta de soldadura sobre las PCBs "
            "a través de un stencil. La calidad del depósito es crítica para "
            "el rendimiento SMT aguas abajo."
        ),
        "parameters": {
            "Squeegee Pressure": {
                "uom": "kg",
                "typical_range": "4 – 10 kg",
                "why_it_matters": "Define la transferencia de pasta al PCB. Muy bajo = poco depósito, muy alto = smearing.",
                "alerts": "Variación >15% entre placas indica desgaste del squeegee o problemas de stencil.",
            },
            "Print Speed": {
                "uom": "mm/s",
                "typical_range": "20 – 150 mm/s",
                "why_it_matters": "Controla cuánto tiempo la pasta fluye a través de las aperturas del stencil.",
                "alerts": "Velocidades muy altas con pasta viscosa = depósito insuficiente.",
            },
            "Separation Speed": {
                "uom": "mm/s",
                "typical_range": "0.5 – 6 mm/s",
                "why_it_matters": "Velocidad con la que el stencil se separa del PCB tras imprimir. Afecta forma del depósito.",
                "alerts": "Demasiado rápido = picos / spiking; demasiado lento = ciclo largo.",
            },
            "Cycle Time": {
                "uom": "s",
                "typical_range": "8 – 25 s",
                "why_it_matters": "Tiempo total por placa. Indicador clave de productividad.",
                "alerts": "Crecimiento sostenido suele indicar atascos de conveyor o problemas de visión.",
            },
            "Stencil Cleaning Cycles": {
                "uom": "ciclos",
                "typical_range": "cada 5 – 20 placas",
                "why_it_matters": "Frecuencia de limpieza del stencil. Excesiva = consumo; baja = bloqueos.",
                "alerts": "Incremento repentino = pasta con problemas de viscosidad o humedad.",
            },
        },
    },
    # ========== PICK & PLACE ==========
    "pick_and_place": {
        "machine_keywords": ["pick and place", "mounter", "chip shooter", "fuji", "yamaha", "asm", "siplace"],
        "description": (
            "Coloca componentes SMD sobre las PCBs con alta precisión y velocidad. "
            "Es el corazón de la línea SMT."
        ),
        "parameters": {
            "CPH (Components Per Hour)": {
                "uom": "cph",
                "typical_range": "15,000 – 120,000 cph",
                "why_it_matters": "Throughput real de la máquina; principal KPI de productividad.",
                "alerts": "Caída >10% vs histórico indica pérdida de nozzles, rechazos o calibración.",
            },
            "Placement Accuracy": {
                "uom": "µm",
                "typical_range": "±30 – ±80 µm",
                "why_it_matters": "Precisión de colocación respecto al fiducial. Crítica para QFN/BGA.",
                "alerts": "Valores fuera de ±50µm para QFN requieren recalibración inmediata.",
            },
            "Pickup Errors": {
                "uom": "count",
                "typical_range": "<0.3% de intentos",
                "why_it_matters": "Componentes no tomados del feeder. Impacta eficiencia.",
                "alerts": "Subidas bruscas = nozzle desgastado, vacío bajo o feeder desalineado.",
            },
            "Vision Fails": {
                "uom": "count",
                "typical_range": "<0.5%",
                "why_it_matters": "Componentes rechazados por visión antes de colocar.",
                "alerts": "Muchas fallas con el mismo componente = feeder sucio o componente mal orientado.",
            },
            "Feeder Errors": {
                "uom": "count",
                "typical_range": "0",
                "why_it_matters": "Errores mecánicos del feeder. Detienen la línea.",
                "alerts": "Requieren intervención inmediata.",
            },
        },
    },
    # ========== REFLOW OVEN ==========
    "reflow_oven": {
        "machine_keywords": ["reflow", "oven", "heller", "btu", "vitronics"],
        "description": (
            "Horno de reflujo que funde la pasta de soldadura siguiendo un "
            "perfil térmico específico para cada producto."
        ),
        "parameters": {
            "Zone Temperature": {
                "uom": "°C",
                "typical_range": "Zonas: 120 – 260 °C según perfil",
                "why_it_matters": "Controla el perfil térmico de reflujo. Base de la calidad de la soldadura.",
                "alerts": "Desviación >±5°C del setpoint = riesgo de defectos (tombstoning, voids).",
            },
            "Conveyor Speed": {
                "uom": "cm/min o mm/s",
                "typical_range": "60 – 140 cm/min",
                "why_it_matters": "Controla el TAL (Time Above Liquidus) y la rampa de temperatura.",
                "alerts": "Cambios no programados afectan todo el perfil.",
            },
            "Exhaust / Cooling": {
                "uom": "Hz o %",
                "typical_range": "según receta",
                "why_it_matters": "Control del enfriamiento post-reflujo; afecta el grano del soldado.",
                "alerts": "Exceso = stress térmico; defecto = soldadura frágil.",
            },
            "Oxygen Level (N2 ovens)": {
                "uom": "ppm",
                "typical_range": "<1000 ppm (típico <500)",
                "why_it_matters": "Atmósfera inerte reduce oxidación de la soldadura.",
                "alerts": ">1500 ppm = pérdida de hermeticidad o N2 bajo.",
            },
        },
    },
    # ========== SPI ==========
    "spi": {
        "machine_keywords": ["spi", "solder paste inspection", "koh young", "cyberoptics"],
        "description": (
            "Inspección 3D de la pasta de soldadura tras la impresora. "
            "Detecta defectos de volumen, altura, área y offset."
        ),
        "parameters": {
            "Volume": {
                "uom": "%",
                "typical_range": "80 – 120% del ideal",
                "why_it_matters": "Volumen de pasta depositado vs el stencil design.",
                "alerts": "<75% insuficiente, >130% excesivo.",
            },
            "Height": {
                "uom": "µm",
                "typical_range": "según espesor de stencil",
                "why_it_matters": "Altura del depósito; indicador clave de la calidad del print.",
                "alerts": "Drift consistente = desgaste de stencil.",
            },
            "Area": {
                "uom": "%",
                "typical_range": "80 – 125%",
                "why_it_matters": "Área del depósito; detecta aperturas bloqueadas.",
                "alerts": "<70% = apertura bloqueada, requiere cleaning.",
            },
            "Offset X/Y": {
                "uom": "µm",
                "typical_range": "<±30 µm",
                "why_it_matters": "Desalineación respecto al pad. Afecta rendimiento.",
                "alerts": "Offset sistemático = recalibrar stencil/PCB.",
            },
            "Reject Rate": {
                "uom": "%",
                "typical_range": "<2%",
                "why_it_matters": "Porcentaje de placas que la SPI marca como NG.",
                "alerts": ">5% = problema en Paste Printer aguas arriba.",
            },
        },
    },
    # ========== AOI ==========
    "aoi": {
        "machine_keywords": ["aoi", "automated optical inspection", "omron", "mirtec", "viscom"],
        "description": (
            "Inspección óptica automática post-reflujo que detecta defectos "
            "como componentes faltantes, mal colocados, soldadura defectuosa."
        ),
        "parameters": {
            "FPY (First Pass Yield)": {
                "uom": "%",
                "typical_range": ">98%",
                "why_it_matters": "Porcentaje de PCBs que pasan AOI sin defectos.",
                "alerts": "<95% = problema sistémico en línea, requiere root-cause.",
            },
            "False Call Rate": {
                "uom": "%",
                "typical_range": "<3%",
                "why_it_matters": "Defectos reportados que en realidad no existen.",
                "alerts": "Alta = programa AOI mal afinado; baja productividad de operadores.",
            },
            "Defect Types": {
                "uom": "count",
                "typical_range": "depende de programa",
                "why_it_matters": "Distribución de tipos de defectos: tombstoning, bridging, missing, etc.",
                "alerts": "Spike de un tipo = investigar proceso relacionado.",
            },
        },
    },
    # ========== GENERAL / SHARED ==========
    "general": {
        "machine_keywords": ["general", "any"],
        "description": "Parámetros comunes que suelen existir en cualquier equipo SMT.",
        "parameters": {
            "Status / State": {
                "uom": "categorical",
                "typical_range": "Running, Idle, Down, Blocked, Starved",
                "why_it_matters": "Estado operativo; base de OEE.",
                "alerts": "Mucho tiempo en Blocked/Starved = problema de línea, no de máquina.",
            },
            "Uptime / Availability": {
                "uom": "%",
                "typical_range": ">85%",
                "why_it_matters": "Fracción del tiempo productivo vs planeado.",
                "alerts": "<80% requiere análisis de paros.",
            },
            "Cycle Time": {
                "uom": "s",
                "typical_range": "depende de producto",
                "why_it_matters": "Tiempo por unidad. Determina throughput.",
                "alerts": "Crecimiento sostenido = desgaste o problema de proceso.",
            },
            "Alarm Count": {
                "uom": "count/hora",
                "typical_range": "<3/hora",
                "why_it_matters": "Frecuencia de alarmas activas.",
                "alerts": "Alta = máquina estresada o mal ajustada.",
            },
        },
    },
}


def guess_category(machine_name: str) -> str:
    """Adivina a qué categoría del KB pertenece una máquina por su nombre."""
    name_l = (machine_name or "").lower()
    for key, entry in PARAMETER_KNOWLEDGE.items():
        if key == "general":
            continue
        for kw in entry["machine_keywords"]:
            if kw in name_l:
                return key
    return "general"


def get_category_info(category: str) -> dict:
    return PARAMETER_KNOWLEDGE.get(category, PARAMETER_KNOWLEDGE["general"])
