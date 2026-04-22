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
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from llm_engine import chat_completion_json, chat_completion, _extract_json

# Optional: statsmodels para ARIMA/SARIMA (pip install statsmodels)
try:
    from statsmodels.tsa.stattools import acf as _sm_acf, pacf as _sm_pacf
    from statsmodels.tsa.ar_model import AutoReg as _AutoReg
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


# ==================================================================
# Sistema prompts
# ==================================================================
_OVERLAY_SYSTEM = """Eres un analista de datos industriales y cientifico de datos experto.
Puedes modificar graficas de series de tiempo de formas ARBITRARIAS: estadistica, modelos predictivos, suavizado, control de calidad, regresiones, etc.
Responde SIEMPRE con JSON valido. Sin texto fuera del JSON.

Esquema de respuesta:
{
  "overlays": [ <lista de capas, ver tipos abajo> ],
  "explanation": "descripcion breve de lo que calculaste"
}

== TIPOS DE OVERLAY DISPONIBLES ==

1. LINEA HORIZONTAL fija:
   {"type": "hline", "value": 123.45, "label": "Media", "color": "#10b981", "dashed": false}

2. BANDA HORIZONTAL:
   {"type": "band", "lower": 90.0, "upper": 110.0, "label": "Rango control", "color": "#f59e0b", "opacity": 0.15}

3. LINEA VERTICAL en tiempo:
   {"type": "vline", "timestamp": "2026-04-20T08:30:00", "label": "Evento", "color": "#ef4444", "dashed": true}

4. SERIE CALCULADA (el mas poderoso — USALO para predicciones, modelos, suavizado, etc.):
   {
     "type": "series",
     "data": [{"t": "2026-04-20T08:00:00", "y": 123.4}, {"t": "2026-04-20T09:00:00", "y": 124.1}, ...],
     "label": "Prediccion 5 dias",
     "color": "#8b5cf6",
     "dashed": true,
     "width": 2
   }
   IMPORTANTE: Calcula TU MISMO los valores de 'data'. Usa regresion lineal, exponencial, media movil, o cualquier modelo que se ajuste mejor a los datos provistos en 'recent_data'.
   Para predicciones futuras: extrapola desde el ultimo timestamp en 'recent_data' usando el intervalo en 'sample_interval_seconds'.

5. TENDENCIA LINEAL (Altair calcula la regresion automaticamente):
   {"type": "trendline", "color": "#3b82f6", "label": "Tendencia"}

6. MEDIA MOVIL (calculada en Python automaticamente):
   {"type": "moving_avg", "window": 10, "color": "#f59e0b", "label": "Media movil"}

7. SUAVIZADO EXPONENCIAL (calculado en Python):
   {"type": "ewm", "span": 5, "color": "#06b6d4", "label": "Suavizado exp."}

== COMO HACER PREDICCIONES ==
Cuando el usuario pida predecir, pronosticar o modelar valores futuros:
1. Analiza los ultimos valores en 'recent_data' para detectar tendencia (subida, bajada, estable, ciclica).
2. Calcula la pendiente de los ultimos puntos con regresion lineal simple: pendiente = (y_ultimo - y_primero) / n_puntos.
3. Genera los puntos futuros: cada punto = ultimo_y + pendiente * i, con timestamp = ultimo_t + sample_interval_seconds * i.
4. Para 5 dias: calcula cuantos intervalos caben en 5 dias = (5*86400) / sample_interval_seconds.
5. Genera la serie con esos puntos y ponla en 'data' del overlay tipo 'series'.
6. Puedes incluir tambien bandas de incertidumbre con tipo 'band' usando +/- 1 sigma como margen.

== REGLAS ==
- Usa SIEMPRE los valores numericos exactos de 'numeric_stats' y 'recent_data' del usuario.
- NUNCA pongas formulas como "mean + 2*std" — haz la aritmetica tu mismo.
- Puedes combinar multiples overlays en el array.
- Si la peticion no es clara, haz lo mas util posible y explica en 'explanation'.

== MODIFICAR OVERLAYS EXISTENTES (PRIORIDAD ALTA) ==
CUANDO el mensaje del usuario haga referencia a algo que ya esta en la grafica
(ej: "la linea AR", "el modelo MA", "la media movil", "esa serie", "ese overlay", "cambia el color",
"hazla punteada", "ponla mas gruesa", "cambia a rojo", "quiero verde", etc.)
DEBES usar "action": "replace_by_label" en lugar de crear un overlay nuevo.

Regla: busca en la lista "existing_overlays" del contexto el overlay cuyo label se parezca mas
al que menciona el usuario (coincidencia parcial, sin importar mayusculas/minusculas).
Solo devuelve los campos que CAMBIAN — Python hace el merge con los datos existentes.
NO re-calcules ni incluyas "data" al modificar una serie existente.

Ejemplos de peticion → respuesta correcta:
  "cambia el color de AR a verde"    → {"type":"series",  "action":"replace_by_label", "label":"AR(5)",    "color":"#10b981"}
  "ponla roja"                       → {"type":"series",  "action":"replace_by_label", "label":"<label del ultimo overlay>", "color":"#ef4444"}
  "la tendencia mas gruesa"          → {"type":"trendline","action":"replace_by_label", "label":"Tendencia","width":3}
  "media movil punteada"             → {"type":"moving_avg","action":"replace_by_label","label":"Media movil","dashed":true}
  "quita el fondo gris"              → {"type":"band",     "action":"replace_by_label", "label":"<label de la banda>", "opacity":0.05}
  "hazla discontinua"                → {"type":"series",  "action":"replace_by_label", "label":"<label>",  "dashed":true}

Si el usuario no especifica cual overlay, usa el ultimo de la lista "existing_overlays".
Si no hay overlays existentes y el usuario pide modificar, CREA uno nuevo apropiado.
"""

_QA_SYSTEM = """Eres un analista de datos industriales SMT experto en estadistica y control de procesos.
Responde preguntas sobre la grafica y sus datos usando SIEMPRE los valores exactos del resumen provisto.
Si los datos incluyen una matriz de correlacion, interpreta los valores de Pearson (rho): >0.7 correlacion fuerte positiva, 0.4-0.7 moderada, <0.2 debil o nula, negativo = correlacion inversa.
Si los datos incluyen un histograma, comenta la distribucion, simetria, outliers y normalidad.
Si los datos son series de tiempo, comenta tendencia, estabilidad, variabilidad y anomalias.
Se conciso y tecnico. Usa los valores reales del resumen. Responde en espanol."""

_FORECAST_SYSTEM = """Eres un experto en series de tiempo industriales (SPC, manufactura SMT, procesos continuos).
Analizas datos historicos y eliges el modelo predictivo mas apropiado.
Responde SOLO con JSON valido, sin texto adicional.

Esquema de respuesta:
{
  "model": "linear" | "polynomial" | "holt" | "seasonal",
  "params": {
    "degree": 2,
    "alpha": 0.3,
    "beta": 0.1,
    "periods_back": 3
  },
  "reasoning": "descripcion breve del patron detectado y por que elegiste este modelo",
  "trend_direction": "subiendo" | "bajando" | "estable" | "ciclico" | "variable"
}

GUIA de seleccion:
- linear: tendencia monotona clara (sube o baja consistentemente sin curvatura)
- polynomial: curvatura notable (aceleracion, desaceleracion, cambio de tendencia en U o invertida)
- holt: tendencia con suavizado (ruido industrial, tendencia suave que cambia gradualmente) -- OPCION PREFERIDA para series industriales
- seasonal: patron ciclico o repetitivo visible en las muestras (turnos de trabajo, ciclos de produccion, temperatura del dia)

PARAMETROS:
- polynomial.degree: 2 para parabolica, 3 para cubica (raramente 4)
- holt.alpha: 0.1 (muy suave) a 0.9 (sigue cada cambio). Defecto: 0.3
- holt.beta: 0.05 (tendencia estable) a 0.3 (tendencia cambiante). Defecto: 0.1
- seasonal.periods_back: cuantos ciclos hacia atras usar para proyectar (1-5)

Analiza 'sample_values' para detectar si los datos suben/bajan/oscilan/tienen curvatura.
"""


_CODE_SYSTEM = """Eres un cientifico de datos senior experto en Python, estadistica, ML, series de tiempo y visualizacion.
Tu tarea: leer el codigo/spec de la grafica, los datos y la peticion del usuario, y devolver Python que genere
overlays y/o sub-graficas para modificar la grafica segun lo pedido.

RESPONDE SIEMPRE CON JSON VALIDO EN ESTE ESQUEMA EXACTO:
{
  "mode": "code" | "overlay",
  "python_code": "<codigo Python completo, SOLO cuando mode=code>",
  "overlays": [ ... ],              // usado solo cuando mode=overlay (cambios simples)
  "explanation": "descripcion breve, 1-3 frases"
}

== CUANDO USAR mode="overlay" ==
Cambios simples y declarativos (color, grosor, dashed, threshold, banda, media movil, EWM, trendline).
Mismos tipos que el sistema OVERLAY (hline/band/vline/series/series_band/trendline/moving_avg/ewm/forecast).
Para modificar un overlay existente: incluye "action":"replace_by_label", "label":"<label>" y los campos a cambiar.

== CUANDO USAR mode="code" ==
Cualquier operacion que requiera calculo custom: Isolation Forest, DBSCAN, K-means, PCA, detección de
anomalías avanzada, algoritmos que no esten en la lista de overlays, transformaciones complejas,
ventanas rodantes customizadas, tests estadisticos, etc.

== CONTRATO DEL CODIGO Python ==
El codigo se ejecuta en un sandbox con estas variables/modulos pre-cargados:
  df            : pandas.DataFrame (datos de la grafica; columnas tipicas: TimeStamp, Value_Num)
  pd, np, alt   : pandas, numpy, altair
  datetime      : modulo datetime de stdlib
  math          : modulo math de stdlib
  sklearn       : scikit-learn (puede ser None si no instalado \u2014 verifica antes de usar)
  statsmodels   : modulo statsmodels (puede ser None)
  scipy         : scipy (puede ser None)
  chart_spec    : dict con la spec Vega-Lite actual de la grafica (informativo)
  result        : dict que DEBES poblar con las claves: "overlays", "sub_charts", "explanation"

REGLAS DEL CODIGO:
1. PUEDES usar "from sklearn.X import Y", "from scipy.X import Y", "from statsmodels.X import Y" — estan permitidos y necesarios.
   Las clases mas comunes ya estan pre-cargadas en el namespace (IsolationForest, KMeans, StandardScaler, PCA, LinearRegression), pero si necesitas otra haz el import.
2. NO uses open(), exec(), eval(), input(), os, sys, subprocess, requests \u2014 prohibido.
3. Para usar sklearn: if sklearn is not None: from sklearn.ensemble import IsolationForest. Si sklearn es None, cae a un metodo alternativo (ej: z-score).
4. Poblar result["overlays"] con dicts del mismo esquema que los overlays JSON.
5. Poblar result["sub_charts"] con dicts {"title":"...", "type":"altair", "chart": <alt.Chart>} o {"type":"table","df":<pd.DataFrame>} o {"type":"markdown","content":"..."}.
6. result["explanation"] \u2014 string en espanol breve.
7. Manejar casos borde: df vacio, columnas faltantes \u2014 con if/return early populando explanation.
8. Para anomalias: marcar los puntos anomalos como un overlay "series" con data=[{"t":ts,"y":val}] y color rojo/naranja, O como puntos sueltos en un sub_chart.

EJEMPLO (Isolation Forest para detectar anomalias — incluye puntos rojos + tabla):
```
if "Value_Num" not in df.columns or len(df) < 10:
    result["explanation"] = "No hay suficientes datos para detectar anomalias."
else:
    d = df.dropna(subset=["Value_Num","TimeStamp"]).copy()
    d["TimeStamp"] = pd.to_datetime(d["TimeStamp"], errors="coerce")
    if sklearn is not None:
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=0.05, random_state=42)
        d["_anom"] = iso.fit_predict(d[["Value_Num"]])
        anom = d[d["_anom"] == -1].sort_values("TimeStamp")
        method = "Isolation Forest"
    else:
        z = (d["Value_Num"] - d["Value_Num"].mean()) / d["Value_Num"].std()
        anom = d[z.abs() > 2.5].sort_values("TimeStamp")
        method = "z-score (|z|>2.5, sklearn no disponible)"
    if not anom.empty:
        # (1) Marcar anomalias como PUNTOS ROJOS discretos sobre la grafica
        result["overlays"].append({
            "type": "points",   # <-- "points" = marcadores discretos (NO linea)
            "data": [{"t": row["TimeStamp"].isoformat(), "y": float(row["Value_Num"])} for _, row in anom.iterrows()],
            "label": f"Anomalias ({len(anom)})",
            "color": "#ef4444",
            "size":  90,
        })
        # (2) Tabla con los valores anomalos
        anom_table = anom[["TimeStamp", "Value_Num"]].copy()
        anom_table.columns = ["Tiempo", "Valor"]
        result["sub_charts"].append({
            "title": f"Tabla de anomalias ({len(anom)} puntos)",
            "type": "table",
            "df": anom_table,
        })
    result["explanation"] = f"Detectadas {len(anom)} anomalias con {method}. Puntos marcados en rojo + tabla debajo."
```

TIPOS DE OVERLAY:
- "points" / "scatter": marcadores discretos (anomalias, outliers, highlights). Ideal para anomaly detection.
- "series": linea conectada (para suavizados, predicciones, medias moviles).
- "hline", "vline", "band", "trendline", "moving_avg", "ewm", "forecast", "series_band".

BUENAS PRACTICAS PROFESIONALES:
- Si detectas anomalias/outliers: usa type="points" en rojo/naranja + tabla en sub_charts con los valores.
- Si haces clustering: colorea puntos por cluster y agrega sub_chart con metricas (silhouette, inertia).
- Si haces descomposicion (STL): agrega sub_charts separados para tendencia/estacional/residual.
- Si haces SPC: agrega hlines para UCL/LCL + tabla de puntos fuera de control.
- Si calculas capability (Cp/Cpk): usa sub_chart tipo markdown con los indices formateados.
- SIEMPRE que tengas datos numericos relevantes, devuelve tambien una tabla (sub_chart type="table").

TRAMPAS COMUNES DE CODIGO — EVITALAS:
- sklearn NO acepta datetime directamente. Convierte TimeStamp a numerico primero:
    x_num = (d["TimeStamp"].astype("int64") // 10**9).values.reshape(-1, 1)  # segundos Unix
  Luego usa x_num en .fit() y .predict().
- Para predecir fechas futuras:
    future_dates = pd.date_range(start=d["TimeStamp"].max(), periods=N, freq=...)
    future_num   = (future_dates.astype("int64") // 10**9).values.reshape(-1, 1)
    future_vals  = model.predict(future_num)
  Luego zip(future_dates, future_vals) para construir los puntos de overlay.
- NUNCA llames .reshape() a DatetimeIndex ni a objetos pd.Timestamp — NO tiene ese metodo.
- Para forecasting lineal simple preferible usa holt o statsmodels.tsa (mejores para series industriales).


== MODIFICAR OVERLAYS EXISTENTES (contexto clave) ==
Recibiras "existing_overlays" con lista de overlays ya visibles (cada uno con su label).
Si el usuario pide cambiar color/estilo/grosor de algo ya visible \u2014 usa mode="overlay" con action="replace_by_label".
Si el usuario no especifica cual, usa el ultimo de existing_overlays.

Devuelve SOLO el JSON final, sin markdown ni texto adicional.
"""


_CODE_FREE_SYSTEM = """Eres un cientifico de datos e ingeniero Python senior con libertad CREATIVA TOTAL.
El usuario invoco el comando @code, esto significa que quiere que uses tu inteligencia general y
tu capacidad generativa para hacer LO QUE PIDA, sin limitarte a overlays predefinidos.

RESPONDE SIEMPRE CON JSON VALIDO EN ESTE ESQUEMA:
{
  "python_code": "<codigo Python completo que implementa lo que pide el usuario>",
  "explanation": "descripcion breve en espanol de lo que hace el codigo"
}

== LIBERTAD TOTAL ==
- Puedes usar CUALQUIER tecnica: ML (sklearn), estadistica (statsmodels, scipy), deep learning,
  procesamiento de senales, deteccion de anomalias, clustering, PCA, UMAP, transformaciones Fourier,
  wavelets, descomposicion de series, SPC, capability indices (Cp/Cpk), pruebas de hipotesis,
  bootstrap, simulaciones Monte Carlo, modelos bayesianos, lo que sea apropiado.
- Puedes crear graficas nuevas completas en sub_charts (histogramas, scatter, heatmaps, boxplots,
  violin, QQ plots, diagramas de control, cartas X-R, cualquier tipo).
- Puedes transformar el df como quieras antes de analizar.
- Si el usuario pide algo creativo, IMPLEMENTALO, no digas que no es posible.

== CONTRATO DEL CODIGO Python ==
Variables disponibles en el namespace:
  df            : pandas.DataFrame con los datos (columnas tipicas: TimeStamp, Value_Num)
  pd, np, alt   : pandas, numpy, altair
  math, datetime: modulos stdlib
  sklearn       : scikit-learn (puede ser None; verifica)
  statsmodels   : statsmodels (puede ser None; verifica)
  scipy         : scipy (puede ser None; verifica)
  chart_spec    : dict con la spec Vega-Lite actual de la grafica
  result        : dict que DEBES poblar con "overlays", "sub_charts", "explanation"

Puedes hacer 'from sklearn.X import Y', imports ampliamente permitidos salvo OS/red/archivos.

POBLAR result:
  result["overlays"]    = [<dicts overlay>]     # superponer a la grafica principal
  result["sub_charts"]  = [<dicts sub-chart>]   # graficas nuevas debajo
  result["explanation"] = "<string espanol>"

Tipos de overlay validos: hline, band, vline, series, series_band, trendline, moving_avg, ewm, forecast.
Para modificar uno existente: {"type":"...", "action":"replace_by_label", "label":"X", <campos>}.

Tipos de sub_chart:
  {"title": "...", "type": "altair",   "chart":   <alt.Chart>}
  {"title": "...", "type": "table",    "df":      <pd.DataFrame>}
  {"title": "...", "type": "markdown", "content": "..."}

== REGLAS ==
1. Maneja df vacio o columnas faltantes con explicacion clara.
2. NO uses open/exec/eval/input/os/sys/requests/subprocess (bloqueados por sandbox).
3. Si sklearn/statsmodels/scipy es None, usa alternativa con numpy/pandas.
4. PIENSA EN GRANDE: si el usuario pide analisis profundo, da overlays + sub_charts + markdown.

Devuelve SOLO el JSON final.
"""


# ==================================================================
# Sandbox de ejecucion de codigo generado por la IA
# ==================================================================
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "bytes": bytes,
    "callable": callable, "chr": chr, "complex": complex, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "format": format, "frozenset": frozenset, "getattr": getattr, "hasattr": hasattr,
    "hash": hash, "hex": hex, "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "list": list, "map": map, "max": max, "min": min,
    "next": next, "object": object, "oct": oct, "ord": ord, "pow": pow, "print": print,
    "range": range, "repr": repr, "reversed": reversed, "round": round, "set": set,
    "slice": slice, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "type": type, "zip": zip, "True": True, "False": False, "None": None,
    # Permitir que sklearn/statsmodels use from X import Y: necesitamos __import__ pero filtrado
}


def _safe_import(name, *args, **kwargs):
    """Permite imports con libertad amplia, bloqueando solo OS/red/procesos/archivos."""
    BLOCKED = {
        "os", "sys", "subprocess", "socket", "shutil", "pathlib",
        "requests", "urllib", "urllib3", "http", "httplib", "httpx",
        "ftplib", "smtplib", "telnetlib", "ssl",
        "ctypes", "cffi", "multiprocessing", "threading", "asyncio",
        "builtins", "importlib", "pickle", "marshal", "shelve",
        "tempfile", "glob", "fileinput", "io", "codecs",
        "webbrowser", "platform", "getpass", "pwd", "grp",
    }
    base = name.split(".")[0]
    if base in BLOCKED:
        raise ImportError(f"Import bloqueado por seguridad (OS/red/archivos): {name}")
    return __import__(name, *args, **kwargs)


def _execute_llm_code(code: str, df, chart_spec: dict | None = None) -> dict:
    """Ejecuta codigo Python generado por la IA en un sandbox restringido.
    Retorna {"overlays":[], "sub_charts":[], "explanation":"", "error":str|None, "stdout":str}.
    """
    import io, contextlib, math as _math, datetime as _datetime
    # Lazy imports opcionales
    try:
        import sklearn as _sklearn
    except Exception:
        _sklearn = None
    try:
        import statsmodels as _statsmodels
    except Exception:
        _statsmodels = None
    try:
        import scipy as _scipy
    except Exception:
        _scipy = None

    result: dict = {"overlays": [], "sub_charts": [], "explanation": ""}

    safe_builtins = dict(_SAFE_BUILTINS)
    safe_builtins["__import__"] = _safe_import

    # Pre-cargar clases sklearn comunes para que el modelo pueda usarlas directamente
    _preloaded = {}
    if _sklearn is not None:
        try:
            from sklearn.ensemble import IsolationForest as _IF, RandomForestRegressor as _RFR
            from sklearn.cluster import KMeans as _KM, DBSCAN as _DBSCAN
            from sklearn.decomposition import PCA as _PCA
            from sklearn.preprocessing import StandardScaler as _SS
            from sklearn.linear_model import LinearRegression as _LR
            _preloaded = {
                "IsolationForest": _IF, "KMeans": _KM, "DBSCAN": _DBSCAN,
                "PCA": _PCA, "StandardScaler": _SS, "LinearRegression": _LR,
                "RandomForestRegressor": _RFR,
            }
        except Exception:
            pass

    namespace: dict = {
        "__builtins__": safe_builtins,
        "pd": pd, "np": np, "alt": alt,
        "math": _math, "datetime": _datetime,
        "sklearn": _sklearn, "statsmodels": _statsmodels, "scipy": _scipy,
        "df": df.copy() if df is not None else None,
        "chart_spec": chart_spec or {},
        "result": result,
        **_preloaded,
    }

    buf = io.StringIO()
    error_msg: str | None = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(code, namespace)
        # Recuperar result mutado (puede haber sido reasignado)
        new_result = namespace.get("result", result)
        if isinstance(new_result, dict):
            result = new_result
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"

    # Normalizar claves
    result.setdefault("overlays", [])
    result.setdefault("sub_charts", [])
    result.setdefault("explanation", "")
    result["error"]  = error_msg
    result["stdout"] = buf.getvalue()
    return result


# ==================================================================
# Comandos analiticos disponibles (para @ autocomplete)
# ==================================================================
_ANALYSIS_COMMANDS: dict[str, dict] = {
    # ------- Estadístico -------
    "@acf": {
        "label": "📊 ACF",
        "desc": "Funcion de Autocorrelacion — correlacion entre la serie y versiones desfasadas",
        "category": "Estadístico",
        "template": "@acf",
        "keywords": ["acf", "autocorrelac", "correlacion lag"],
    },
    "@pacf": {
        "label": "📊 PACF",
        "desc": "Autocorrelacion Parcial — identifica el orden p para modelo AR",
        "category": "Estadístico",
        "template": "@pacf",
        "keywords": ["pacf", "autocorrelacion parcial", "parcial autocorr"],
    },
    # ------- Modelos predictivos -------
    "@ar": {
        "label": "🔢 AR",
        "desc": "Modelo Autorregresivo AR(p) — predice basado en valores anteriores",
        "category": "Modelo",
        "template": "@ar predice 1 dia adelante",
        "keywords": ["autorregres", "ar(", "modelo ar"],
    },
    "@ma": {
        "label": "〰️ MA",
        "desc": "Promedio Movil MA(q) — modelo basado en errores pasados",
        "category": "Modelo",
        "template": "Ajusta modelo de Promedio Movil MA y predice el comportamiento",
        "keywords": ["ma(", "modelo ma ", "moving average modelo"],
    },
    "@arima": {
        "label": "📈 ARIMA",
        "desc": "ARIMA(p,d,q) — integra diferenciacion para series no estacionarias",
        "category": "Modelo",
        "template": "@arima predice 3 dias adelante",
        "keywords": ["arima"],
    },
    "@sarima": {
        "label": "🌊 SARIMA",
        "desc": "SARIMA estacional — captura ciclos de turnos o produccion",
        "category": "Modelo",
        "template": "@sarima predice 1 semana adelante",
        "keywords": ["sarima", "sarimax", "estacional arima"],
    },
    "@nonlinear": {
        "label": "🔀 No Lineal",
        "desc": "Modelo de regimen dual TAR — detecta dos estados de operacion",
        "category": "Modelo",
        "template": "@nonlinear predice con modelo TAR",
        "keywords": ["no lineal", "nonlinear", "tar ", "regimen dual"],
    },
    "@holt": {
        "label": "📉 Holt",
        "desc": "Suavizado exponencial doble — nivel + tendencia",
        "category": "Modelo",
        "template": "@holt predice 3 dias adelante",
        "keywords": [],
    },
    "@seasonal": {
        "label": "🔄 Estacional FFT",
        "desc": "Prediccion estacional por FFT — detecta ciclos dominantes",
        "category": "Modelo",
        "template": "@seasonal predice 1 semana adelante",
        "keywords": [],
    },
    # ------- Validación -------
    "@compare": {
        "label": "🏆 Comparar Modelos",
        "desc": "Ajusta varios modelos y compara RMSE / MAE en datos de prueba",
        "category": "Validación",
        "template": "@compare",
        "keywords": ["comparar modelos", "comparacion modelos", "rmse", "mae", "mejor modelo"],
    },
    "@residuals": {
        "label": "🔍 Residuos",
        "desc": "Analiza residuos del modelo — independencia y homocedasticidad",
        "category": "Validación",
        "template": "@residuals",
        "keywords": ["residuos", "residual", "homocedasticidad", "independencia residuos"],
    },
    # ------- Visual -------
    "@sigma": {
        "label": "📐 Sigma",
        "desc": "Bandas de control ±2σ o ±3σ",
        "category": "Visual",
        "template": "Dibuja limites de control a 2 sigma",
        "keywords": [],
    },
    "@movavg": {
        "label": "〰️ Media Movil",
        "desc": "Promedio movil de N puntos",
        "category": "Visual",
        "template": "Agrega media movil de 20 puntos",
        "keywords": [],
    },
    "@ewm": {
        "label": "📊 EWM",
        "desc": "Suavizado exponencial ponderado",
        "category": "Visual",
        "template": "Suaviza la senal con EWM alpha 0.3",
        "keywords": [],
    },
    "@code": {
        "label": "🧠 Code (IA libre)",
        "desc": "Da libertad total a la IA: genera Python para cualquier analisis/modelo/visualizacion",
        "category": "🧠 IA Libre",
        "template": "@code ",
        "keywords": [],
    },
}


# ==================================================================
# Construccion de overlays
# ==================================================================
def _parse_horizon_seconds(prompt: str) -> float | None:
    """Extrae el horizonte temporal de un prompt.
    Soporta: dias, horas, minutos, semanas. Retorna segundos, o None si no detecta.
    """
    import re
    p = prompt.lower()
    # Buscar patrones como "5 dias", "2 horas", "30 minutos", "1 semana"
    patterns = [
        (r"(\d+(?:[.,]\d+)?)\s*(?:dia|día|day)s?", 86400),
        (r"(\d+(?:[.,]\d+)?)\s*(?:hora|hour|hr)s?", 3600),
        (r"(\d+(?:[.,]\d+)?)\s*(?:minuto|minute|min)s?", 60),
        (r"(\d+(?:[.,]\d+)?)\s*(?:semana|week|wk)s?", 7 * 86400),
        (r"(\d+(?:[.,]\d+)?)\s*(?:mes|month)(?:es)?", 30 * 86400),
    ]
    for regex, unit_sec in patterns:
        m = re.search(regex, p)
        if m:
            try:
                return float(m.group(1).replace(",", ".")) * unit_sec
            except ValueError:
                continue
    return None


def _build_forecast_overlay(prompt: str, df: pd.DataFrame | None) -> list[dict] | None:
    """Calcula una prediccion lineal (regresion) server-side sobre df.
    Retorna overlays de tipo 'series' (prediccion) + 'band' (incertidumbre).
    """
    if df is None or df.empty:
        return None
    if "Value_Num" not in df.columns or "TimeStamp" not in df.columns:
        return None

    d = df.dropna(subset=["Value_Num", "TimeStamp"]).copy()
    if len(d) < 5:
        return None
    d["TimeStamp"] = pd.to_datetime(d["TimeStamp"], errors="coerce")
    d = d.dropna(subset=["TimeStamp"]).sort_values("TimeStamp")
    if len(d) < 5:
        return None

    # Horizonte en segundos (por defecto: igual al rango historico, o 1 dia si es mas)
    horizon_s = _parse_horizon_seconds(prompt)
    t0 = d["TimeStamp"].iloc[0]
    t_last = d["TimeStamp"].iloc[-1]
    hist_span_s = (t_last - t0).total_seconds()
    if horizon_s is None:
        horizon_s = max(hist_span_s, 86400)

    # Regresion lineal: y = a*x + b  (x = segundos desde t0)
    secs = (d["TimeStamp"] - t0).dt.total_seconds().values
    yval = d["Value_Num"].values
    try:
        slope, intercept = np.polyfit(secs, yval, 1)
    except Exception:
        return None

    # Residuos para incertidumbre
    predictions_hist = slope * secs + intercept
    residuals = yval - predictions_hist
    sigma = float(np.std(residuals, ddof=0))

    # Generar 60 puntos futuros distribuidos uniformemente en el horizonte
    n_pts = 60
    step_s = horizon_s / n_pts
    last_sec = secs[-1]
    future_points = []
    future_lower = []
    future_upper = []
    for i in range(1, n_pts + 1):
        x = last_sec + step_s * i
        y = slope * x + intercept
        t = t0 + pd.Timedelta(seconds=x)
        # Intervalo de confianza se expande con la distancia desde los datos
        widening = 1.0 + (step_s * i) / max(hist_span_s, 1.0) * 0.5
        ci = 1.96 * sigma * widening
        future_points.append({"t": t.isoformat(), "y": round(float(y), 4)})
        future_lower.append({"t": t.isoformat(), "y": round(float(y - ci), 4)})
        future_upper.append({"t": t.isoformat(), "y": round(float(y + ci), 4)})

    # Incluir el ultimo punto real para continuidad visual
    anchor = {"t": t_last.isoformat(), "y": round(float(yval[-1]), 4)}
    future_points.insert(0, anchor)

    # Etiqueta del horizonte en formato legible
    if horizon_s >= 86400 * 7:
        horizon_label = f"{horizon_s / (86400 * 7):g} sem"
    elif horizon_s >= 86400:
        horizon_label = f"{horizon_s / 86400:g} d"
    elif horizon_s >= 3600:
        horizon_label = f"{horizon_s / 3600:g} h"
    else:
        horizon_label = f"{horizon_s / 60:g} min"

    trend_dir = "sube" if slope > 0 else "baja" if slope < 0 else "estable"
    return [
        # Banda de incertidumbre (IC 95%)
        {
            "type": "series_band",
            "lower": future_lower,
            "upper": future_upper,
            "color": "#8b5cf6",
            "opacity": 0.15,
            "label": f"IC 95%",
        },
        # Linea predictiva
        {
            "type": "series",
            "data": future_points,
            "color": "#8b5cf6",
            "dashed": True,
            "width": 2.5,
            "label": f"Prediccion {horizon_label} ({trend_dir})",
        },
        # Separador vertical en el ultimo dato real
        {
            "type": "vline",
            "timestamp": t_last.isoformat(),
            "color": "#8b5cf6",
            "dashed": True,
            "label": "Inicio prediccion",
        },
    ]


def _local_overlay_fallback(prompt: str, summary: dict, df: pd.DataFrame | None = None) -> list[dict] | None:
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

    # ---- Modelos predictivos ----
    wants_forecast = any(w in p for w in ("predic", "forecast", "pronostic", "proyecc", "futuro", "proximo", "next "))
    wants_ewm = any(w in p for w in ("exponencial", "ewm", "ema", "suavizado"))
    wants_ma  = any(w in p for w in ("media movil", "moving avg", "rolling", "promedio movil", "suaviza", "smooth"))

    num_match = re.search(r"(\d+)\s*(?:puntos?|pts?|ventana|window|periodos?|muestras?)", p)
    num_val   = int(num_match.group(1)) if num_match else None

    # ---- Prediccion con regresion lineal (server-side) ----
    if wants_forecast:
        pred_overlay = _build_forecast_overlay(prompt, df)
        if pred_overlay is not None:
            overlays.extend(pred_overlay)
    if wants_ewm:
        span = num_val or 5
        overlays.append({"type": "ewm", "span": span, "color": "#06b6d4", "label": f"Suavizado exp.(span={span})"})
    elif wants_ma:
        window = num_val or 10
        overlays.append({"type": "moving_avg", "window": window, "color": "#f59e0b", "label": f"Media movil ({window}pts)"})

    return overlays if overlays else None


def _build_summary(df: pd.DataFrame | None, context: dict | None) -> dict:
    """Resumen compacto de estadisticas para enviar al LLM."""
    summary = dict(context or {})
    if df is None or df.empty:
        return summary

    cols = set(df.columns)

    # ---- Matriz de correlacion (param_a, param_b, corr) ----
    if {"param_a", "param_b", "corr"}.issubset(cols):
        valid = df.dropna(subset=["corr"])
        # Pares unicos (sin duplicados ni diagonal)
        pairs = (
            valid[valid["param_a"] != valid["param_b"]]
            .copy()
            .assign(key=lambda d: d.apply(
                lambda r: tuple(sorted([r["param_a"], r["param_b"]])), axis=1
            ))
            .drop_duplicates(subset="key")
            .drop(columns="key")
            .sort_values("corr", key=abs, ascending=False)
        )
        summary["correlation_pairs"] = [
            {"A": row["param_a"], "B": row["param_b"], "rho": round(float(row["corr"]), 4)}
            for _, row in pairs.iterrows()
        ]
        top_pos = pairs[pairs["corr"] > 0].head(5)
        top_neg = pairs[pairs["corr"] < 0].head(5)
        if not top_pos.empty:
            summary["top_positive_correlations"] = [
                {"A": r["param_a"], "B": r["param_b"], "rho": round(float(r["corr"]), 4)}
                for _, r in top_pos.iterrows()
            ]
        if not top_neg.empty:
            summary["top_negative_correlations"] = [
                {"A": r["param_a"], "B": r["param_b"], "rho": round(float(r["corr"]), 4)}
                for _, r in top_neg.iterrows()
            ]
        params = sorted(set(valid["param_a"].tolist() + valid["param_b"].tolist()))
        summary["parameters"] = params
        summary["n_params"] = len(params)
        return summary

    # ---- Serie de tiempo (TimeStamp + Value_Num) ----
    if "TimeStamp" in cols:
        ts = pd.to_datetime(df["TimeStamp"], errors="coerce").dropna()
        if not ts.empty:
            summary["time_start"] = ts.min().isoformat()
            summary["time_end"] = ts.max().isoformat()
            summary["n_points"] = int(len(ts))

    if "Value_Num" in cols:
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
                "cv_pct": round(float(v.std(ddof=0) / v.mean() * 100), 2) if v.mean() != 0 else None,
            }
            # Muestra distribucion para histogramas
            if context and context.get("type") == "histogram":
                counts, edges = pd.cut(v, bins=min(20, max(5, len(v)//10)), retbins=True)
                freq = counts.value_counts(sort=False)
                summary["histogram_bins"] = [
                    {"range": f"{edges[i]:.3g}-{edges[i+1]:.3g}", "count": int(c)}
                    for i, c in enumerate(freq)
                ]

        # recent_data: ultimos 80 puntos (o submuestreados a 80) para que el LLM pueda calcular modelos
        if "TimeStamp" in cols:
            d = df.dropna(subset=["Value_Num", "TimeStamp"]).copy()
            d["TimeStamp"] = pd.to_datetime(d["TimeStamp"], errors="coerce")
            d = d.sort_values("TimeStamp")
            if len(d) > 80:
                # Submuestrear uniformemente a 80 puntos
                idx = np.linspace(0, len(d) - 1, 80, dtype=int)
                d = d.iloc[idx]
            if not d.empty:
                summary["recent_data"] = [
                    {"t": row["TimeStamp"].isoformat(), "y": round(float(row["Value_Num"]), 4)}
                    for _, row in d.iterrows()
                ]
                # Intervalo tipico entre muestras en segundos
                if len(d) >= 2:
                    intervals = d["TimeStamp"].diff().dt.total_seconds().dropna()
                    summary["sample_interval_seconds"] = round(float(intervals.median()), 1)

    # ---- Categoricos ----
    if "Value_Str" in cols and "Value_Num" not in cols:
        v = df["Value_Str"].dropna().astype(str)
        if not v.empty:
            top = v.value_counts().head(5).to_dict()
            summary["categorical_top"] = {str(k): int(cnt) for k, cnt in top.items()}
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

            elif ov_type in ("series", "points", "scatter") and "data" in ov:
                # Serie calculada por el LLM: [{"t": ISO_str, "y": float}, ...]
                # type="series"  -> linea continua (min 2 puntos)
                # type="points"  -> marcadores discretos (anomalias, outliers); cualquier cantidad >= 1
                # type="scatter" -> alias de points
                raw_data = ov["data"]
                as_points = ov_type in ("points", "scatter") or ov.get("mark") in ("point", "circle", "dot")
                min_pts = 1 if as_points else 2
                if isinstance(raw_data, list) and len(raw_data) >= min_pts:
                    series_df = pd.DataFrame(raw_data)
                    series_df["t"] = pd.to_datetime(series_df["t"], errors="coerce")
                    series_df["y"] = pd.to_numeric(series_df["y"], errors="coerce")
                    series_df = series_df.dropna()
                    if not series_df.empty:
                        stroke_w = float(ov.get("width", 2))
                        if as_points:
                            size = float(ov.get("size", 80))
                            s_layer = alt.Chart(series_df).mark_point(
                                color=color, filled=True, size=size,
                                opacity=float(ov.get("opacity", 0.9)),
                            ).encode(
                                x="t:T", y="y:Q",
                                tooltip=[alt.Tooltip("t:T", title="Tiempo"),
                                         alt.Tooltip("y:Q", title="Valor", format=".4f")],
                            )
                        else:
                            s_layer = alt.Chart(series_df).mark_line(
                                color=color, strokeDash=stroke_dash, strokeWidth=stroke_w, opacity=0.9
                            ).encode(x="t:T", y="y:Q")
                        layers.append(s_layer)
                        if label:
                            last = series_df.iloc[-1]
                            txt_df = pd.DataFrame({"x": [last["t"]], "y": [last["y"]]})
                            txt = alt.Chart(txt_df).mark_text(
                                align="left", dx=6, dy=-6, color=color,
                                fontSize=11, fontWeight="bold"
                            ).encode(x="x:T", y="y:Q", text=alt.value(label))
                            layers.append(txt)

            elif ov_type == "series_band" and "lower" in ov and "upper" in ov:
                # Banda de incertidumbre: lower y upper son listas [{"t", "y"}, ...]
                lo_data = ov["lower"]
                hi_data = ov["upper"]
                if (isinstance(lo_data, list) and isinstance(hi_data, list)
                        and len(lo_data) == len(hi_data) and len(lo_data) >= 2):
                    band_df = pd.DataFrame({
                        "t": [pd.to_datetime(p["t"], errors="coerce") for p in lo_data],
                        "lower": [float(p["y"]) for p in lo_data],
                        "upper": [float(p["y"]) for p in hi_data],
                    }).dropna()
                    if not band_df.empty:
                        opacity = float(ov.get("opacity", 0.15))
                        area = alt.Chart(band_df).mark_area(
                            color=color, opacity=opacity
                        ).encode(x="t:T", y="lower:Q", y2="upper:Q")
                        layers.append(area)

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

            elif ov_type == "moving_avg" and df is not None and "Value_Num" in df.columns and "TimeStamp" in df.columns:
                d = df.dropna(subset=["Value_Num", "TimeStamp"]).copy()
                d["TimeStamp"] = pd.to_datetime(d["TimeStamp"], errors="coerce")
                d = d.sort_values("TimeStamp")
                window = max(2, int(ov.get("window", 10)))
                if len(d) >= window:
                    d["_ma"] = d["Value_Num"].rolling(window=window, min_periods=2, center=True).mean()
                    plot_d = d.dropna(subset=["_ma"])[["TimeStamp", "_ma"]].rename(columns={"_ma": "Value_Num"})
                    if not plot_d.empty:
                        ma_line = alt.Chart(plot_d).mark_line(
                            color=color, strokeWidth=2, opacity=0.9
                        ).encode(x="TimeStamp:T", y="Value_Num:Q")
                        layers.append(ma_line)
                        if label:
                            last = plot_d.iloc[-1]
                            txt_df = pd.DataFrame({"x": [last["TimeStamp"]], "y": [last["Value_Num"]]})
                            txt = alt.Chart(txt_df).mark_text(
                                align="left", dx=6, dy=-6, color=color, fontSize=11, fontWeight="bold"
                            ).encode(x="x:T", y="y:Q", text=alt.value(label))
                            layers.append(txt)

            elif ov_type == "ewm" and df is not None and "Value_Num" in df.columns and "TimeStamp" in df.columns:
                d = df.dropna(subset=["Value_Num", "TimeStamp"]).copy()
                d["TimeStamp"] = pd.to_datetime(d["TimeStamp"], errors="coerce")
                d = d.sort_values("TimeStamp")
                span = max(2, int(ov.get("span", 5)))
                if len(d) >= 3:
                    d["_ewm"] = d["Value_Num"].ewm(span=span, adjust=False).mean()
                    plot_d = d[["TimeStamp", "_ewm"]].rename(columns={"_ewm": "Value_Num"})
                    ewm_line = alt.Chart(plot_d).mark_line(
                        color=color, strokeWidth=2, opacity=0.9
                    ).encode(x="TimeStamp:T", y="Value_Num:Q")
                    layers.append(ewm_line)
                    if label:
                        last = plot_d.iloc[-1]
                        txt_df = pd.DataFrame({"x": [last["TimeStamp"]], "y": [last["Value_Num"]]})
                        txt = alt.Chart(txt_df).mark_text(
                            align="left", dx=6, dy=-6, color=color, fontSize=11, fontWeight="bold"
                        ).encode(x="x:T", y="y:Q", text=alt.value(label))
                        layers.append(txt)

            elif ov_type == "forecast" and df is not None and "Value_Num" in df.columns and "TimeStamp" in df.columns:
                d = df.dropna(subset=["Value_Num", "TimeStamp"]).copy()
                d["TimeStamp"] = pd.to_datetime(d["TimeStamp"], errors="coerce")
                d = d.sort_values("TimeStamp")
                if len(d) >= 5:
                    t0   = d["TimeStamp"].min()
                    secs = (d["TimeStamp"] - t0).dt.total_seconds().values
                    yval = d["Value_Num"].values
                    coeffs = np.polyfit(secs, yval, 1)          # [slope, intercept]
                    periods = max(5, int(ov.get("periods", 20)))
                    intervals = np.diff(secs)
                    step = float(np.median(intervals)) if len(intervals) > 0 else 60.0
                    last_t   = d["TimeStamp"].max()
                    last_sec = secs[-1]
                    future_ts = [last_t + pd.Timedelta(seconds=step * (i + 1)) for i in range(periods)]
                    future_y  = [float(np.polyval(coeffs, last_sec + step * (i + 1))) for i in range(periods)]
                    # Continuity: include last real point
                    anchor = d[["TimeStamp", "Value_Num"]].tail(1)
                    fcst_df = pd.DataFrame({"TimeStamp": future_ts, "Value_Num": future_y})
                    all_df  = pd.concat([anchor, fcst_df], ignore_index=True)
                    fc_line = alt.Chart(all_df).mark_line(
                        color=color, strokeDash=[8, 4], strokeWidth=2, opacity=0.85
                    ).encode(x="TimeStamp:T", y="Value_Num:Q")
                    layers.append(fc_line)
                    # Vertical separator at forecast boundary
                    sep_df = pd.DataFrame({"x": [last_t]})
                    sep = alt.Chart(sep_df).mark_rule(
                        color=color, strokeDash=[4, 4], strokeWidth=1, opacity=0.4
                    ).encode(x="x:T")
                    layers.append(sep)
                    if label:
                        end_df = pd.DataFrame({"x": [future_ts[-1]], "y": [future_y[-1]]})
                        txt = alt.Chart(end_df).mark_text(
                            align="left", dx=6, dy=-6, color=color, fontSize=11, fontWeight="bold"
                        ).encode(x="x:T", y="y:Q", text=alt.value(label))
                        layers.append(txt)

        except Exception:
            continue

    if len(layers) == 1:
        return base_chart
    return alt.layer(*layers).resolve_scale(y="shared")


# ==================================================================
# Modelos predictivos server-side
# ==================================================================
def _holt_smooth(
    values: np.ndarray, alpha: float, beta: float
) -> tuple[np.ndarray, float, float]:
    """Holt double exponential smoothing. Returns (smoothed_array, last_level, last_trend)."""
    alpha = max(0.05, min(0.95, float(alpha)))
    beta  = max(0.01, min(0.50, float(beta)))
    level = float(values[0])
    trend = float(values[1] - values[0]) if len(values) > 1 else 0.0
    smoothed = [level]
    for y in values[1:]:
        prev = level
        level = alpha * float(y) + (1 - alpha) * (level + trend)
        trend = beta  * (level - prev) + (1 - beta) * trend
        smoothed.append(level)
    return np.array(smoothed), level, trend


def _compute_model_series(
    spec: dict, df: pd.DataFrame, horizon_s: float
) -> list[dict]:
    """Ejecuta el modelo predictivo elegido por el LLM y retorna overlays."""
    model  = str(spec.get("model", "holt")).lower()
    params = spec.get("params") or {}

    if df is None or df.empty:
        return []
    if "Value_Num" not in df.columns or "TimeStamp" not in df.columns:
        return []

    d = df.dropna(subset=["Value_Num", "TimeStamp"]).copy()
    d["TimeStamp"] = pd.to_datetime(d["TimeStamp"], errors="coerce")
    d = d.dropna(subset=["TimeStamp"]).sort_values("TimeStamp")
    if len(d) < 5:
        return []

    t0     = d["TimeStamp"].iloc[0]
    t_last = d["TimeStamp"].iloc[-1]
    secs   = (d["TimeStamp"] - t0).dt.total_seconds().values
    yval   = d["Value_Num"].values.astype(float)

    diffs  = np.diff(secs)
    step_s = float(np.median(diffs)) if len(diffs) > 0 else 60.0
    n_pts  = max(30, min(150, int(horizon_s / max(step_s, 1))))
    last_s = secs[-1]
    future_secs = np.array([last_s + step_s * (i + 1) for i in range(n_pts)])
    future_ts   = [t0 + pd.Timedelta(seconds=float(x)) for x in future_secs]

    # ---------- Compute future_y per model ----------
    if model == "polynomial":
        degree = max(2, min(4, int(params.get("degree", 2))))
        try:
            coeffs  = np.polyfit(secs, yval, degree)
            future_y = np.polyval(coeffs, future_secs)
            sigma   = float(np.std(yval - np.polyval(coeffs, secs), ddof=0))
        except Exception:
            coeffs  = np.polyfit(secs, yval, 1)
            future_y = np.polyval(coeffs, future_secs)
            sigma   = float(np.std(yval - np.polyval(coeffs, secs), ddof=0))

    elif model == "holt":
        alpha = float(params.get("alpha", 0.3))
        beta  = float(params.get("beta",  0.1))
        _, level, trend = _holt_smooth(yval, alpha, beta)
        # trend is in units of y per sample; convert to y per second
        avg_interval = float(np.mean(diffs)) if len(diffs) > 0 else step_s
        trend_per_s  = trend / max(avg_interval, 1.0)
        future_y = np.array([level + trend_per_s * (step_s * (i + 1)) for i in range(n_pts)])
        # Residuals from smoothed signal
        smoothed, _, _ = _holt_smooth(yval, alpha, beta)
        sigma = float(np.std(yval - smoothed, ddof=0))

    elif model == "seasonal":
        slope, intercept = np.polyfit(secs, yval, 1)
        residuals = yval - (slope * secs + intercept)
        n = len(residuals)
        period_s = None
        if n >= 8:
            freqs   = np.fft.rfftfreq(n, d=step_s)
            magnitudes = np.abs(np.fft.rfft(residuals))
            if len(magnitudes) > 1:
                dom_idx = int(np.argmax(magnitudes[1:]) + 1)
                dom_freq = freqs[dom_idx]
                period_s = 1.0 / dom_freq if dom_freq > 0 else None

        future_y = []
        periods_back = int(params.get("periods_back", 2))
        for i in range(n_pts):
            x = future_secs[i]
            y_trend = slope * x + intercept
            if period_s and period_s > step_s * 3:
                # Use average residual at same phase from last N periods
                phase = x % period_s
                tol   = step_s * 1.5
                mask  = np.abs(secs % period_s - phase) <= tol
                # Restrict to last periods_back periods
                cutoff_s = last_s - period_s * periods_back
                mask = mask & (secs >= cutoff_s)
                y_seasonal = float(np.mean(residuals[mask])) if mask.any() else 0.0
            else:
                y_seasonal = 0.0
            future_y.append(y_trend + y_seasonal)
        future_y = np.array(future_y)
        sigma = float(np.std(residuals, ddof=0))

    else:  # linear (default)
        coeffs   = np.polyfit(secs, yval, 1)
        future_y = np.polyval(coeffs, future_secs)
        sigma    = float(np.std(yval - np.polyval(coeffs, secs), ddof=0))

    # ---------- Build overlays ----------
    anchor_pt = {"t": t_last.isoformat(), "y": round(float(yval[-1]), 4)}
    data_pts  = [anchor_pt] + [
        {"t": ts.isoformat(), "y": round(float(y), 4)}
        for ts, y in zip(future_ts, future_y)
    ]
    lower_pts = [anchor_pt]
    upper_pts = [anchor_pt]
    for i, (ts, y) in enumerate(zip(future_ts, future_y)):
        widening = 1.0 + (i / max(n_pts, 1)) * 0.6
        ci = 1.96 * sigma * widening
        lower_pts.append({"t": ts.isoformat(), "y": round(float(y - ci), 4)})
        upper_pts.append({"t": ts.isoformat(), "y": round(float(y + ci), 4)})

    return [
        {"type": "series_band", "lower": lower_pts, "upper": upper_pts,
         "color": "#8b5cf6", "opacity": 0.13},
        {"type": "series", "data": data_pts, "color": "#8b5cf6",
         "dashed": True, "width": 2.5, "label": f"Prediccion ({model})"},
        {"type": "vline", "timestamp": t_last.isoformat(),
         "color": "#8b5cf6", "dashed": True, "label": ""},
    ]


def _llm_guided_forecast(
    prompt: str, df: pd.DataFrame, context: dict | None, summary: dict
) -> tuple[list[dict], str]:
    """
    LLM analiza el patron de los datos y elige el mejor modelo.
    Python ejecuta el calculo real. Retorna (overlays, explanation_str).
    """
    horizon_s = _parse_horizon_seconds(prompt)
    if horizon_s is None:
        # Inferir horizonte: mismo span historico, maximo 7 dias
        d = df.dropna(subset=["Value_Num", "TimeStamp"]).copy() if df is not None else pd.DataFrame()
        if not d.empty and "TimeStamp" in d.columns:
            d["TimeStamp"] = pd.to_datetime(d["TimeStamp"], errors="coerce")
            d = d.dropna(subset=["TimeStamp"]).sort_values("TimeStamp")
            if len(d) >= 2:
                span_s = (d["TimeStamp"].iloc[-1] - d["TimeStamp"].iloc[0]).total_seconds()
                horizon_s = min(max(span_s, 3600), 7 * 86400)
        if horizon_s is None:
            horizon_s = 86400

    # Muestra compacta para el LLM (primeros 5, medio 5, ultimos 10)
    recent = summary.get("recent_data", [])
    n = len(recent)
    if n > 20:
        idxs = sorted(set(
            list(range(5)) +
            list(range(n // 2 - 2, n // 2 + 3)) +
            list(range(max(0, n - 10), n))
        ))
        sample = [recent[i] for i in idxs if i < n]
    else:
        sample = recent

    llm_input = {
        "numeric_stats":          summary.get("numeric_stats", {}),
        "sample_interval_seconds": summary.get("sample_interval_seconds"),
        "n_points":                summary.get("n_points"),
        "time_span": {"start": summary.get("time_start"), "end": summary.get("time_end")},
        "sample_values":           sample,
    }

    if horizon_s >= 86400 * 7:
        h_label = f"{horizon_s / (86400*7):.1g} semana(s)"
    elif horizon_s >= 86400:
        h_label = f"{horizon_s / 86400:.1g} dia(s)"
    elif horizon_s >= 3600:
        h_label = f"{horizon_s / 3600:.1g} hora(s)"
    else:
        h_label = f"{horizon_s / 60:.1g} minuto(s)"

    user_msg = (
        f"Peticion: {prompt}\n"
        f"Horizonte de prediccion solicitado: {h_label}\n\n"
        f"Datos:\n{json.dumps(llm_input, ensure_ascii=False, indent=2)}"
    )

    try:
        spec = chat_completion_json(_FORECAST_SYSTEM, user_msg, temperature=0.1)
        if "error" in spec or "model" not in spec:
            spec = {"model": "holt", "params": {"alpha": 0.3, "beta": 0.1},
                    "reasoning": "Modelo Holt por defecto.", "trend_direction": "variable"}
    except Exception:
        spec = {"model": "holt", "params": {"alpha": 0.3, "beta": 0.1},
                "reasoning": "Modelo Holt por defecto.", "trend_direction": "variable"}

    overlays    = _compute_model_series(spec, df, horizon_s)
    model_name  = spec.get("model", "holt")
    reasoning   = spec.get("reasoning", "")
    trend_dir   = spec.get("trend_direction", "")
    explanation = f"Prediccion {h_label} · modelo {model_name} · tendencia: {trend_dir}"
    return overlays, explanation, reasoning


# ==================================================================
# Analisis estadisticos avanzados (ACF, PACF, AR, MA, ARIMA …)
# ==================================================================

def _prep_ts_data(df) -> dict | None:
    """Extrae y limpia la serie de tiempo del DataFrame. Retorna dict con arrays clave o None."""
    if df is None or df.empty:
        return None
    if "Value_Num" not in df.columns or "TimeStamp" not in df.columns:
        return None
    d = df.dropna(subset=["Value_Num", "TimeStamp"]).copy()
    d["TimeStamp"] = pd.to_datetime(d["TimeStamp"], errors="coerce")
    d = d.dropna(subset=["TimeStamp"]).sort_values("TimeStamp")
    if len(d) < 10:
        return None
    t0    = d["TimeStamp"].iloc[0]
    secs  = (d["TimeStamp"] - t0).dt.total_seconds().values
    yval  = d["Value_Num"].values.astype(float)
    diffs = np.diff(secs)
    step_s = float(np.median(diffs)) if len(diffs) > 0 else 60.0
    return {"d": d, "secs": secs, "yval": yval, "step_s": step_s,
            "t0": t0, "ts": d["TimeStamp"]}


def _compute_acf_values(values: np.ndarray, nlags: int) -> tuple[np.ndarray, float]:
    n = len(values)
    x = values - np.mean(values)
    denom = float(np.dot(x, x))
    if denom == 0:
        return np.zeros(nlags + 1), 1.96 / np.sqrt(max(n, 1))
    acf_vals = [1.0]
    for k in range(1, nlags + 1):
        c = float(np.dot(x[:n - k], x[k:])) / denom
        acf_vals.append(c)
    ci = 1.96 / np.sqrt(n)
    return np.array(acf_vals), ci


def _compute_pacf_values(values: np.ndarray, nlags: int) -> tuple[np.ndarray, float]:
    acf_vals, ci = _compute_acf_values(values, nlags)
    pacf_vals = [1.0]
    if nlags >= 1:
        pacf_vals.append(float(acf_vals[1]))
    phi = np.array([acf_vals[1]]) if nlags >= 1 else np.array([])
    var  = max(1.0 - acf_vals[1] ** 2, 1e-10) if nlags >= 1 else 1.0
    for k in range(2, nlags + 1):
        numerator   = acf_vals[k] - float(np.dot(phi, acf_vals[k - 1:0:-1]))
        phi_k       = numerator / max(var, 1e-10)
        phi_k       = max(-0.99, min(0.99, phi_k))
        new_phi     = phi - phi_k * phi[::-1]
        phi         = np.append(new_phi, phi_k)
        var         = max(var * (1.0 - phi_k ** 2), 1e-10)
        pacf_vals.append(float(phi_k))
    ci = 1.96 / np.sqrt(len(values))
    return np.array(pacf_vals), ci


def _corr_bar_chart(corr_vals: np.ndarray, ci: float, title: str) -> alt.Chart:
    nlags = len(corr_vals) - 1
    data  = pd.DataFrame({"lag": list(range(nlags + 1)), "corr": corr_vals.tolist()})
    bars  = (
        alt.Chart(data)
        .mark_bar(size=6, color="#3b82f6")
        .encode(
            x=alt.X("lag:Q", title="Lag"),
            y=alt.Y("corr:Q", title="Correlacion",
                    scale=alt.Scale(domain=[-1.1, 1.1])),
            color=alt.condition(
                alt.datum.corr > 0,
                alt.value("#3b82f6"), alt.value("#ef4444"),
            ),
            tooltip=[alt.Tooltip("lag:Q"), alt.Tooltip("corr:Q", format=".4f")],
        )
    )
    ci_df    = pd.DataFrame({"y": [ci, -ci]})
    ci_lines = (
        alt.Chart(ci_df)
        .mark_rule(color="#f59e0b", strokeDash=[4, 2])
        .encode(y="y:Q")
    )
    zero = (
        alt.Chart(pd.DataFrame({"y": [0.0]}))
        .mark_rule(color="#6b7280", size=1)
        .encode(y="y:Q")
    )
    return alt.layer(ci_lines, zero, bars).properties(title=title, height=220)


def _forecast_overlays_from_arrays(
    future_y: np.ndarray, future_ts: list, yval: np.ndarray,
    t_last, residuals: np.ndarray, label: str, color: str = "#8b5cf6",
) -> list[dict]:
    """Construye overlays series + series_band + vline desde arrays de forecast."""
    n_pts = len(future_y)
    sigma = float(np.std(residuals, ddof=0)) if len(residuals) > 1 else float(np.std(yval) * 0.1)
    anchor = {"t": t_last.isoformat(), "y": round(float(yval[-1]), 4)}
    data_pts = [anchor] + [
        {"t": ts.isoformat(), "y": round(float(y), 4)}
        for ts, y in zip(future_ts, future_y)
    ]
    lower_pts = [anchor]
    upper_pts = [anchor]
    for i, (ts, y) in enumerate(zip(future_ts, future_y)):
        ci = 1.96 * sigma * (1.0 + (i / max(n_pts, 1)) * 0.6)
        lower_pts.append({"t": ts.isoformat(), "y": round(float(y - ci), 4)})
        upper_pts.append({"t": ts.isoformat(), "y": round(float(y + ci), 4)})
    return [
        {"type": "series_band", "lower": lower_pts, "upper": upper_pts,
         "color": color, "opacity": 0.13},
        {"type": "series", "data": data_pts, "color": color,
         "dashed": True, "width": 2.5, "label": label},
        {"type": "vline", "timestamp": t_last.isoformat(),
         "color": color, "dashed": True, "label": ""},
    ]


def _build_future_timestamps(t0, last_s: float, step_s: float, n_pts: int):
    return [t0 + pd.Timedelta(seconds=last_s + step_s * (i + 1)) for i in range(n_pts)]


# ---- ACF ----
def _analysis_acf(df, nlags: int = 40) -> dict:
    ts = _prep_ts_data(df)
    if ts is None:
        return {"error": "Se necesitan datos numericos con timestamps para ACF."}
    acf_vals, ci = _compute_acf_values(ts["yval"], nlags)
    chart = _corr_bar_chart(acf_vals, ci, f"ACF — Autocorrelacion (n={len(ts['yval'])})")
    sig = int(np.sum(np.abs(acf_vals[1:]) > ci))
    return {
        "overlays": [],
        "sub_charts": [{"title": "📊 ACF — Funcion de Autocorrelacion",
                         "type": "altair", "chart": chart}],
        "explanation": (
            f"ACF calculada con {nlags} lags. "
            f"**{sig}** lag(s) superan el intervalo de confianza (IC = ±{ci:.3f}). "
            "Lags significativos sugieren estructura temporal aprovechable para modelos AR/MA."
        ),
    }


# ---- PACF ----
def _analysis_pacf(df, nlags: int = 40) -> dict:
    ts = _prep_ts_data(df)
    if ts is None:
        return {"error": "Se necesitan datos numericos con timestamps para PACF."}
    pacf_vals, ci = _compute_pacf_values(ts["yval"], nlags)
    chart = _corr_bar_chart(pacf_vals, ci, f"PACF — Autocorrelacion Parcial (n={len(ts['yval'])})")
    sig_lags = [k for k, v in enumerate(pacf_vals[1:], 1) if abs(v) > ci]
    p_suggest = sig_lags[-1] if sig_lags else 1
    return {
        "overlays": [],
        "sub_charts": [{"title": "📊 PACF — Autocorrelacion Parcial",
                         "type": "altair", "chart": chart}],
        "explanation": (
            f"PACF calculada con {nlags} lags. "
            f"Lags significativos: {sig_lags[:10]}. "
            f"Orden p sugerido para AR: **{p_suggest}**."
        ),
    }


# ---- AR ----
def _analysis_ar(df, horizon_s: float) -> dict:
    ts = _prep_ts_data(df)
    if ts is None:
        return {"error": "Se necesitan datos numericos con timestamps para AR."}
    yval, secs, step_s, t0 = ts["yval"], ts["secs"], ts["step_s"], ts["t0"]
    n = len(yval)

    # Seleccionar orden p via PACF
    max_p = min(20, n // 5)
    pacf_vals, ci = _compute_pacf_values(yval, max_p)
    sig_lags = [k for k, v in enumerate(pacf_vals[1:], 1) if abs(v) > ci]
    p = sig_lags[-1] if sig_lags else min(5, max_p)
    p = max(1, min(p, max_p))

    if _HAS_STATSMODELS:
        try:
            mdl = _AutoReg(yval, lags=p, old_names=False).fit()
            residuals = mdl.resid
            n_pts = max(30, min(120, int(horizon_s / max(step_s, 1))))
            future_y = mdl.forecast(steps=n_pts)
            future_ts = _build_future_timestamps(t0, float(secs[-1]), step_s, n_pts)
            info = f"AR({p}) via statsmodels — AIC={mdl.aic:.2f}, BIC={mdl.bic:.2f}"
        except Exception as e:
            _HAS_STATSMODELS_local = False
            info = f"AR({p}) fallback numpy (statsmodels error: {e})"
            residuals, future_y, future_ts = None, None, None
    else:
        _HAS_STATSMODELS_local = False
        info = f"AR({p}) numpy"
        residuals, future_y, future_ts = None, None, None

    if future_y is None:
        # Numpy fallback
        X = np.stack([yval[i: n - p + i] for i in range(p)], axis=1)
        y_train = yval[p:]
        coeffs, _, _, _ = np.linalg.lstsq(X, y_train, rcond=None)
        residuals = y_train - X @ coeffs
        n_pts = max(30, min(120, int(horizon_s / max(step_s, 1))))
        buf = list(yval[-p:])
        future_y_list = []
        for _ in range(n_pts):
            yh = float(np.dot(buf, coeffs))
            future_y_list.append(yh)
            buf = buf[1:] + [yh]
        future_y = np.array(future_y_list)
        future_ts = _build_future_timestamps(t0, float(secs[-1]), step_s, n_pts)

    t_last = ts["ts"].iloc[-1]
    overlays = _forecast_overlays_from_arrays(
        future_y, future_ts, yval, t_last, residuals, f"AR({p})", "#6366f1"
    )
    return {"overlays": overlays, "sub_charts": [], "explanation": info}


# ---- MA ----
def _analysis_ma(df, horizon_s: float) -> dict:
    ts = _prep_ts_data(df)
    if ts is None:
        return {"error": "Se necesitan datos numericos con timestamps para MA."}
    yval, secs, step_s, t0 = ts["yval"], ts["secs"], ts["step_s"], ts["t0"]

    if _HAS_STATSMODELS:
        try:
            # Detect q via ACF
            acf_v, ci = _compute_acf_values(yval, min(20, len(yval) // 5))
            sig_q = [k for k, v in enumerate(acf_v[1:], 1) if abs(v) > ci]
            q = sig_q[-1] if sig_q else 2
            q = max(1, min(q, 10))
            mdl = _ARIMA(yval, order=(0, 0, q)).fit()
            residuals = mdl.resid
            n_pts = max(30, min(120, int(horizon_s / max(step_s, 1))))
            fc = mdl.forecast(steps=n_pts)
            future_ts = _build_future_timestamps(t0, float(secs[-1]), step_s, n_pts)
            t_last = ts["ts"].iloc[-1]
            overlays = _forecast_overlays_from_arrays(
                fc, future_ts, yval, t_last, residuals, f"MA({q})", "#0ea5e9"
            )
            info = f"MA({q}) via statsmodels — AIC={mdl.aic:.2f}"
        except Exception as e:
            return {"error": f"MA fallido: {e}"}
    else:
        # Approximation: EWM-based MA-like forecast
        span = min(10, len(yval) // 5)
        smoothed = pd.Series(yval).ewm(span=span, adjust=False).mean().values
        residuals = yval - smoothed
        n_pts = max(30, min(120, int(horizon_s / max(step_s, 1))))
        future_y = np.full(n_pts, float(smoothed[-1]))  # constant forecast
        future_ts = _build_future_timestamps(t0, float(secs[-1]), step_s, n_pts)
        t_last = ts["ts"].iloc[-1]
        overlays = _forecast_overlays_from_arrays(
            future_y, future_ts, yval, t_last, residuals, f"MA(approx,span={span})", "#0ea5e9"
        )
        info = "MA aproximado con EWM (instala statsmodels para MA exacto)"

    return {"overlays": overlays, "sub_charts": [], "explanation": info}


# ---- ARIMA ----
def _analysis_arima(df, horizon_s: float) -> dict:
    ts = _prep_ts_data(df)
    if ts is None:
        return {"error": "Se necesitan datos numericos con timestamps para ARIMA."}
    yval, secs, step_s, t0 = ts["yval"], ts["secs"], ts["step_s"], ts["t0"]

    if not _HAS_STATSMODELS:
        return {"error": "ARIMA requiere statsmodels. Instala con: pip install statsmodels"}

    try:
        # Auto-select orders: try (2,1,2), (1,1,1), (1,0,1), fall back to (1,1,0)
        best_aic = float("inf")
        best_mdl = None
        best_order = (1, 1, 1)
        for order in [(1, 1, 1), (2, 1, 2), (1, 0, 1), (1, 1, 0), (0, 1, 1)]:
            try:
                m = _ARIMA(yval, order=order).fit()
                if m.aic < best_aic:
                    best_aic  = m.aic
                    best_mdl  = m
                    best_order = order
            except Exception:
                pass
        if best_mdl is None:
            return {"error": "No se pudo ajustar ninguna variante ARIMA."}

        n_pts = max(30, min(120, int(horizon_s / max(step_s, 1))))
        fc    = best_mdl.forecast(steps=n_pts)
        residuals = best_mdl.resid
        future_ts = _build_future_timestamps(t0, float(secs[-1]), step_s, n_pts)
        t_last = ts["ts"].iloc[-1]
        overlays = _forecast_overlays_from_arrays(
            fc, future_ts, yval, t_last, residuals,
            f"ARIMA{best_order}", "#f59e0b"
        )
        info = f"ARIMA{best_order} — AIC={best_aic:.2f} (mejor de 5 variantes)"
        return {"overlays": overlays, "sub_charts": [], "explanation": info}
    except Exception as e:
        return {"error": f"ARIMA fallido: {e}"}


# ---- SARIMA ----
def _analysis_sarima(df, horizon_s: float) -> dict:
    ts = _prep_ts_data(df)
    if ts is None:
        return {"error": "Se necesitan datos numericos con timestamps para SARIMA."}
    yval, secs, step_s, t0 = ts["yval"], ts["secs"], ts["step_s"], ts["t0"]

    if not _HAS_STATSMODELS:
        return {"error": "SARIMA requiere statsmodels. Instala con: pip install statsmodels"}

    try:
        # Infer seasonal period from FFT
        n = len(yval)
        trend_coeffs = np.polyfit(np.arange(n), yval, 1)
        detrended = yval - np.polyval(trend_coeffs, np.arange(n))
        freqs = np.fft.rfftfreq(n)
        mags  = np.abs(np.fft.rfft(detrended))
        dom_freq = freqs[int(np.argmax(mags[1:])) + 1] if len(mags) > 1 else 0
        s = int(round(1.0 / dom_freq)) if dom_freq > 0 else 12
        s = max(2, min(s, max(2, n // 4)))

        best_aic = float("inf")
        best_mdl = None
        best_order = ((1, 1, 1), (1, 0, 1, s))
        for pdq in [(1, 1, 1), (1, 0, 1), (0, 1, 1)]:
            for PDQ in [(1, 0, 1, s), (0, 1, 1, s)]:
                try:
                    m = _SARIMAX(yval, order=pdq, seasonal_order=PDQ,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False).fit(disp=False)
                    if m.aic < best_aic:
                        best_aic   = m.aic
                        best_mdl   = m
                        best_order = (pdq, PDQ)
                except Exception:
                    pass
        if best_mdl is None:
            return {"error": f"No se pudo ajustar SARIMA con periodo s={s}."}

        n_pts = max(30, min(120, int(horizon_s / max(step_s, 1))))
        fc    = best_mdl.forecast(steps=n_pts)
        residuals = best_mdl.resid
        future_ts = _build_future_timestamps(t0, float(secs[-1]), step_s, n_pts)
        t_last = ts["ts"].iloc[-1]
        overlays = _forecast_overlays_from_arrays(
            fc, future_ts, yval, t_last, residuals,
            f"SARIMA{best_order[0]}x{best_order[1]}", "#10b981"
        )
        info = f"SARIMA{best_order[0]}x{best_order[1]} — periodo s={s} — AIC={best_aic:.2f}"
        return {"overlays": overlays, "sub_charts": [], "explanation": info}
    except Exception as e:
        return {"error": f"SARIMA fallido: {e}"}


# ---- No Lineal (TAR — Threshold Autoregressive) ----
def _analysis_nonlinear(df, horizon_s: float) -> dict:
    ts = _prep_ts_data(df)
    if ts is None:
        return {"error": "Se necesitan datos numericos con timestamps para modelo no lineal."}
    yval, secs, step_s, t0 = ts["yval"], ts["secs"], ts["step_s"], ts["t0"]
    n = len(yval)
    threshold = float(np.median(yval))
    p = max(1, min(5, n // 10))

    # Fit two AR models: high and low regime
    def fit_ar(y_all, mask, p_):
        if np.sum(mask[p_:]) < p_ + 2:
            return None, None
        rows = [i for i in range(p_, len(y_all)) if mask[i]]
        if len(rows) < p_ + 2:
            return None, None
        X = np.stack([y_all[i - p_:i] for i in rows], axis=0)
        y_reg = np.array([y_all[i] for i in rows])
        c, _, _, _ = np.linalg.lstsq(X, y_reg, rcond=None)
        res = y_reg - X @ c
        return c, res

    high_mask = yval >= threshold
    low_mask  = yval < threshold
    c_high, r_high = fit_ar(yval, high_mask, p)
    c_low,  r_low  = fit_ar(yval, low_mask,  p)

    if c_high is None or c_low is None:
        return {"error": "No hay suficientes datos en cada regimen para TAR."}

    residuals = np.concatenate([r_high, r_low])
    buf = list(yval[-p:])
    n_pts = max(30, min(120, int(horizon_s / max(step_s, 1))))
    future_y_list = []
    for _ in range(n_pts):
        regime = "high" if buf[-1] >= threshold else "low"
        coeffs = c_high if regime == "high" else c_low
        yh = float(np.dot(buf, coeffs))
        future_y_list.append(yh)
        buf = buf[1:] + [yh]

    future_y  = np.array(future_y_list)
    future_ts = _build_future_timestamps(t0, float(secs[-1]), step_s, n_pts)
    t_last = ts["ts"].iloc[-1]
    overlays = _forecast_overlays_from_arrays(
        future_y, future_ts, yval, t_last, residuals,
        f"TAR(p={p}, umbral={threshold:.2f})", "#ec4899"
    )
    info = (
        f"Modelo no lineal TAR con p={p}, umbral={threshold:.2f}. "
        f"Regimen alto (≥{threshold:.2f}): {int(np.sum(high_mask))} puntos. "
        f"Regimen bajo (<{threshold:.2f}): {int(np.sum(low_mask))} puntos."
    )
    return {"overlays": overlays, "sub_charts": [], "explanation": info}


# ---- Comparacion de modelos ----
def _analysis_compare(df) -> dict:
    ts = _prep_ts_data(df)
    if ts is None:
        return {"error": "Se necesitan datos numericos con timestamps para comparar modelos."}
    yval, secs, step_s = ts["yval"], ts["secs"], ts["step_s"]
    n = len(yval)
    split = max(int(n * 0.8), n - 30)
    y_train, y_test = yval[:split], yval[split:]
    s_train, s_test = secs[:split], secs[split:]
    if len(y_test) < 3:
        return {"error": "No hay suficientes datos de prueba (necesita ≥ 15 puntos total)."}

    results = []

    def _rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def _mae(a, b):
        return float(np.mean(np.abs(a - b)))

    n_test = len(y_test)

    # 1. Linear
    try:
        c = np.polyfit(s_train, y_train, 1)
        pred = np.polyval(c, s_test)
        results.append({"Modelo": "Linear", "RMSE": _rmse(y_test, pred), "MAE": _mae(y_test, pred)})
    except Exception:
        pass

    # 2. Polynomial (degree=2)
    try:
        c = np.polyfit(s_train, y_train, 2)
        pred = np.polyval(c, s_test)
        results.append({"Modelo": "Polynomial(2)", "RMSE": _rmse(y_test, pred), "MAE": _mae(y_test, pred)})
    except Exception:
        pass

    # 3. Holt
    try:
        _, level, trend_holt = _holt_smooth(y_train, 0.3, 0.1)
        avg_int = float(np.mean(np.diff(s_train))) if len(s_train) > 1 else step_s
        trend_ps = trend_holt / max(avg_int, 1.0)
        pred = np.array([level + trend_ps * (step_s * (i + 1)) for i in range(n_test)])
        results.append({"Modelo": "Holt", "RMSE": _rmse(y_test, pred), "MAE": _mae(y_test, pred)})
    except Exception:
        pass

    # 4. AR (numpy)
    try:
        p = min(5, len(y_train) // 5)
        X = np.stack([y_train[i: len(y_train) - p + i] for i in range(p)], axis=1)
        y_reg = y_train[p:]
        coeffs_ar, _, _, _ = np.linalg.lstsq(X, y_reg, rcond=None)
        buf = list(y_train[-p:])
        pred_ar = []
        for _ in range(n_test):
            yh = float(np.dot(buf, coeffs_ar))
            pred_ar.append(yh)
            buf = buf[1:] + [yh]
        pred = np.array(pred_ar)
        results.append({"Modelo": f"AR({p})", "RMSE": _rmse(y_test, pred), "MAE": _mae(y_test, pred)})
    except Exception:
        pass

    # 5. ARIMA (statsmodels)
    if _HAS_STATSMODELS:
        try:
            m = _ARIMA(y_train, order=(1, 1, 1)).fit()
            pred = m.forecast(steps=n_test)
            results.append({"Modelo": "ARIMA(1,1,1)", "RMSE": _rmse(y_test, pred), "MAE": _mae(y_test, pred)})
        except Exception:
            pass

    if not results:
        return {"error": "No se pudo ajustar ningun modelo."}

    df_res = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    df_res["RMSE"] = df_res["RMSE"].round(4)
    df_res["MAE"]  = df_res["MAE"].round(4)
    df_res.index  = range(1, len(df_res) + 1)
    best = df_res.iloc[0]["Modelo"]

    # Build bar chart for RMSE
    bar_data = df_res.copy()
    bar_data["rank"] = ["🥇 " + bar_data.iloc[0]["Modelo"]] + [
        str(r) for r in bar_data.iloc[1:]["Modelo"]
    ]
    chart = (
        alt.Chart(bar_data)
        .mark_bar()
        .encode(
            y=alt.Y("Modelo:N", sort="-x", title="Modelo"),
            x=alt.X("RMSE:Q", title="RMSE (menor = mejor)"),
            color=alt.condition(
                alt.datum.Modelo == best,
                alt.value("#10b981"), alt.value("#6b7280"),
            ),
            tooltip=["Modelo:N",
                     alt.Tooltip("RMSE:Q", format=".4f"),
                     alt.Tooltip("MAE:Q", format=".4f")],
        )
        .properties(title=f"Comparacion de modelos — {n_test} puntos de prueba", height=220)
    )
    explanation = (
        f"**Mejor modelo: {best}** (RMSE={df_res.iloc[0]['RMSE']:.4f}, "
        f"MAE={df_res.iloc[0]['MAE']:.4f}). "
        f"Comparados {len(df_res)} modelos en {n_test} puntos de prueba (ultimo 20% de los datos)."
    )
    return {
        "overlays": [],
        "sub_charts": [
            {"title": "🏆 Comparacion de Modelos (RMSE/MAE)",
             "type": "altair", "chart": chart},
            {"title": "📋 Tabla de Resultados",
             "type": "table", "df": df_res},
        ],
        "explanation": explanation,
    }


# ---- Analisis Residual ----
def _analysis_residuals(df) -> dict:
    ts = _prep_ts_data(df)
    if ts is None:
        return {"error": "Se necesitan datos numericos con timestamps para analisis residual."}
    yval, secs, step_s, t0 = ts["yval"], ts["secs"], ts["step_s"], ts["t0"]
    n = len(yval)
    p = min(5, n // 10)

    # Fit AR(p) to get residuals
    if p >= 1:
        X = np.stack([yval[i: n - p + i] for i in range(p)], axis=1)
        y_reg = yval[p:]
        coeffs, _, _, _ = np.linalg.lstsq(X, y_reg, rcond=None)
        residuals = y_reg - X @ coeffs
        t_res = ts["ts"].iloc[p:].reset_index(drop=True)
        info_model = f"Residuos de AR({p})"
    else:
        coeffs_lin = np.polyfit(secs, yval, 1)
        residuals  = yval - np.polyval(coeffs_lin, secs)
        t_res = ts["ts"].reset_index(drop=True)
        info_model = "Residuos de regresion lineal"

    # Residuals over time
    df_resid = pd.DataFrame({"t": t_res.values, "residual": residuals})
    res_chart = (
        alt.Chart(df_resid)
        .mark_line(color="#6366f1", size=1.5)
        .encode(
            x=alt.X("t:T", title="Tiempo"),
            y=alt.Y("residual:Q", title="Residuo"),
            tooltip=[alt.Tooltip("t:T"), alt.Tooltip("residual:Q", format=".4f")],
        )
        .properties(title=f"Residuos en el tiempo ({info_model})", height=200)
    )
    zero_line = (
        alt.Chart(pd.DataFrame({"y": [0.0]}))
        .mark_rule(color="#ef4444", strokeDash=[4, 2])
        .encode(y="y:Q")
    )
    res_chart = alt.layer(res_chart, zero_line)

    # Rolling variance (homocedasticidad)
    window = max(5, len(residuals) // 10)
    roll_var = pd.Series(residuals).rolling(window).var().values
    df_rv = pd.DataFrame({"t": t_res.values, "var": roll_var})
    rv_chart = (
        alt.Chart(df_rv.dropna())
        .mark_area(color="#f59e0b", opacity=0.4)
        .encode(
            x=alt.X("t:T", title="Tiempo"),
            y=alt.Y("var:Q", title="Varianza movil"),
        )
        .properties(title=f"Varianza movil de residuos (ventana={window}) — homocedasticidad", height=160)
    )

    # ACF of residuals
    acf_r, ci_r = _compute_acf_values(residuals, min(30, len(residuals) // 3))
    acf_chart = _corr_bar_chart(acf_r, ci_r, "ACF de residuos — debe ser ruido blanco")

    sig_r = [k for k, v in enumerate(acf_r[1:], 1) if abs(v) > ci_r]
    if not sig_r:
        residual_verdict = "✅ Residuos parecen ruido blanco (sin autocorrelacion significativa)."
    else:
        residual_verdict = (
            f"⚠️ Residuos tienen autocorrelacion en lags {sig_r[:5]} — "
            "el modelo puede no capturar toda la estructura."
        )

    # Basic stats
    mean_r = float(np.mean(residuals))
    std_r  = float(np.std(residuals))

    explanation = (
        f"{info_model}. "
        f"Media={mean_r:.4f}, Desv.Std={std_r:.4f}. "
        f"{residual_verdict}"
    )
    return {
        "overlays": [],
        "sub_charts": [
            {"title": "🔍 Residuos en el tiempo", "type": "altair", "chart": res_chart},
            {"title": "📊 Varianza movil (homocedasticidad)", "type": "altair", "chart": rv_chart},
            {"title": "📊 ACF de residuos", "type": "altair", "chart": acf_chart},
        ],
        "explanation": explanation,
    }


# ---- Dispatcher principal ----
def _detect_analysis_command(prompt: str) -> str | None:
    """Detecta que comando de analisis se solicita en el prompt. Retorna clave '@cmd' o None."""
    p = prompt.lower().strip()

    # Explicit @command
    for cmd in _ANALYSIS_COMMANDS:
        kw = cmd[1:]  # "acf", "pacf", "ar", ...
        if p.startswith(f"@{kw}") or f" @{kw}" in p:
            return cmd

    # Keyword matching
    for cmd, info in _ANALYSIS_COMMANDS.items():
        for kw in info.get("keywords", []):
            if kw and kw in p:
                return cmd

    return None


def _dispatch_analysis_command(prompt: str, df, context) -> dict:
    """Ejecuta el analisis correspondiente al comando detectado."""
    cmd = _detect_analysis_command(prompt)
    if cmd is None:
        return {}

    horizon_s = _parse_horizon_seconds(prompt) or 86400

    if cmd == "@acf":
        nlags = 40
        for tok in prompt.split():
            try:
                v = int(tok)
                if 5 <= v <= 200:
                    nlags = v
                    break
            except ValueError:
                pass
        return _analysis_acf(df, nlags)

    if cmd == "@pacf":
        nlags = 40
        for tok in prompt.split():
            try:
                v = int(tok)
                if 5 <= v <= 200:
                    nlags = v
                    break
            except ValueError:
                pass
        return _analysis_pacf(df, nlags)

    if cmd == "@ar":
        return _analysis_ar(df, horizon_s)

    if cmd == "@ma":
        return _analysis_ma(df, horizon_s)

    if cmd == "@arima":
        return _analysis_arima(df, horizon_s)

    if cmd == "@sarima":
        return _analysis_sarima(df, horizon_s)

    if cmd == "@nonlinear":
        return _analysis_nonlinear(df, horizon_s)

    if cmd in ("@holt", "@seasonal"):
        # Route to LLM-guided forecast with forced model
        spec = {
            "model": "holt" if cmd == "@holt" else "seasonal",
            "params": {"alpha": 0.3, "beta": 0.1, "periods_back": 2},
            "reasoning": f"Forzado por comando {cmd}",
            "trend_direction": "variable",
        }
        overlays = _compute_model_series(spec, df, horizon_s)
        if horizon_s >= 86400:
            h = f"{horizon_s/86400:.1g}d"
        else:
            h = f"{horizon_s/3600:.1g}h"
        return {
            "overlays": overlays, "sub_charts": [],
            "explanation": f"Prediccion {h} con modelo {spec['model']}",
        }

    if cmd == "@compare":
        return _analysis_compare(df)

    if cmd == "@residuals":
        return _analysis_residuals(df)

    # Visual overlays (@sigma, @movavg, @ewm) — pass through to local fallback
    return {}


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
            # scope=fragment: solo recarga este bloque, evita scroll al fondo
            st.rerun(scope="fragment")

    st.altair_chart(final_chart, use_container_width=True)

    # -------- Resumen + codigo generado por la IA (si hubo @code reciente) --------
    last_ai = state.get("_last_ai_code")
    if last_ai:
        msg = last_ai.get("summary", "")
        if msg:
            st.success(msg)
        with st.expander("📜 Codigo generado por la IA", expanded=False):
            st.code(last_ai.get("code", ""), language="python")
            if last_ai.get("stdout"):
                st.caption("**Salida del codigo:**")
                st.code(last_ai["stdout"], language="text")
        col_a, col_b = st.columns([1, 6])
        with col_a:
            if st.button("🗑 Ocultar codigo", key=f"{state_key}_hide_code",
                         help="Ocultar el codigo y el resumen"):
                state["_last_ai_code"] = None
                st.rerun(scope="fragment")

    # -------- Sub-charts (ACF, PACF, residuos, comparacion) --------
    if state.get("sub_charts"):
        for sub in state["sub_charts"]:
            with st.expander(sub["title"], expanded=True):
                if sub["type"] == "altair":
                    st.altair_chart(sub["chart"], use_container_width=True)
                elif sub["type"] == "table":
                    st.dataframe(sub["df"], use_container_width=True)
                elif sub["type"] == "markdown":
                    st.markdown(sub.get("content", ""))
        if st.button("🗑 Limpiar análisis", key=f"{state_key}_clear_sub",
                     help="Ocultar graficas de analisis estadistico"):
            state["sub_charts"] = []
            st.rerun(scope="fragment")

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
                    st.rerun(scope="fragment")
            elif current_mode == "💬 Preguntar sobre los datos" and has_history:
                if st.button("🗑", key=f"{state_key}_clear", use_container_width=True,
                             help="Limpiar historial del chat"):
                    state["history"] = []
                    st.rerun(scope="fragment")

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
            "sub_charts": [],
            "_show_cmds": False,
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

# Palabras que indican intencion predictiva compleja (LLM-guided)
_COMPLEX_PREDICT_WORDS = (
    "comportamiento", "patron", "modelo pred", "estacional", "ciclico",
    "en base al", "segun los datos", "analiza", "sugier", "recomiend",
    "proximo", "siguiente", "proyect", "extrapolac",
)
# Palabras que indican prediccion simple (local linear, sin LLM)
_SIMPLE_PREDICT_WORDS = (
    "predic", "forecast", "pronostic", "futuro",
)


def _exec_analysis_in_place(
    template: str, df, context: dict | None, state: dict, state_key: str
) -> bool:
    """
    Ejecuta un comando de analisis directamente (desde click en sugerencia).
    Retorna True si fue manejado, False si debe pasar por el pipeline de texto.
    """
    result = _dispatch_analysis_command(template, df, context)
    if not result:
        return False  # comando visual (@sigma etc.) — pasa al pipeline normal
    if "error" in result:
        st.error(result["error"])
        if not _HAS_STATSMODELS and "statsmodels" in result.get("error", ""):
            st.code("pip install statsmodels", language="bash")
        return True
    new_overlays = result.get("overlays", [])
    new_sub      = result.get("sub_charts", [])
    explanation  = result.get("explanation", "")
    if new_overlays:
        state["overlays"].extend(new_overlays)
    if new_sub:
        state["sub_charts"] = state.get("sub_charts", []) + new_sub
    state["last_explanation"] = explanation
    state["_show_cmds"] = False
    pkey = f"{state_key}_prompt_val"
    if pkey in st.session_state:
        del st.session_state[pkey]
    return True


def _render_cmd_suggestions(
    state: dict, state_key: str, df, context, filter_str: str = ""
) -> None:
    """
    Renderiza el panel de sugerencias de comandos @.
    filter_str: lo que el usuario escribio despues del @, para filtrar en tiempo real.
    """
    categories: dict[str, list] = {}
    for cmd, info in _ANALYSIS_COMMANDS.items():
        if filter_str:
            searchable = (
                cmd[1:] + " " + info["label"] + " " + info["desc"] + " " +
                " ".join(info.get("keywords", []))
            ).lower()
            if filter_str not in searchable:
                continue
        cat = info.get("category", "General")
        categories.setdefault(cat, []).append((cmd, info))

    if not categories:
        st.caption(f"_Sin resultados para `@{filter_str}` — prueba: acf, arima, compare, residuals…_")
        return

    header = (
        f"Sugerencias para **`@{filter_str}`** — haz clic para ejecutar:"
        if filter_str else
        "**Comandos @ disponibles** — haz clic para ejecutar directamente en la gráfica:"
    )
    with st.container(border=True):
        st.caption(header)
        for cat, cmds in sorted(categories.items()):
            st.markdown(
                f"<small style='color:#9ca3af'><b>{cat}</b></small>",
                unsafe_allow_html=True,
            )
            n_cols = min(len(cmds), 3)
            cols = st.columns(n_cols)
            for i, (cmd, info) in enumerate(cmds):
                with cols[i % n_cols]:
                    if st.button(
                        info["label"],
                        key=f"{state_key}_cmd_{cmd}",
                        help=info["desc"],
                        use_container_width=True,
                    ):
                        executed = _exec_analysis_in_place(
                            info["template"], df, context, state, state_key
                        )
                        if executed:
                            st.rerun()
                        else:
                            # Comando visual: pre-poblar input y ejecutar al aplicar
                            state["_pending_prompt"] = info["template"]
                            state["_show_cmds"] = False
                            st.rerun(scope="fragment")
                    # Descripcion breve visible bajo el boton
                    short = info["desc"][:62] + "…" if len(info["desc"]) > 62 else info["desc"]
                    st.caption(f"_{short}_")


def _inject_autocomplete_js(input_key: str, submit_btn_key: str) -> None:
    """
    Inyecta JS via components.html (iframe same-origin) para:
    1. Dropdown morado con sugerencias @ en tiempo real
    2. Color/tipografia morada del input cuando empieza con '@'
    3. Submit con Enter
    NOTA: st.markdown(unsafe_allow_html) NO ejecuta <script> en React —
          components.html() ejecuta JS correctamente y puede acceder a
          window.parent.document por ser same-origin.
    """
    import streamlit.components.v1 as _stcomp

    cmds_json = json.dumps([
        {"cmd": cmd, "label": info["label"], "desc": info["desc"],
         "cat": info.get("category", "General")}
        for cmd, info in _ANALYSIS_COMMANDS.items()
    ])

    html = f"""<!DOCTYPE html>
<html><head><style>body{{margin:0;padding:0;overflow:hidden;}}</style></head>
<body><script>
(function() {{
  const CMDS = {cmds_json};
  let dropdown = null;

  function findInput() {{
    try {{
      const inputs = window.parent.document.querySelectorAll('input[type="text"]');
      for (const inp of inputs) {{
        if (inp.placeholder && inp.placeholder.includes('@')) return inp;
      }}
    }} catch(e) {{}}
    return null;
  }}

  function findApplyBtn() {{
    try {{
      const btns = window.parent.document.querySelectorAll('button');
      for (const b of btns) {{
        if (b.innerText && b.innerText.includes('Aplicar')) return b;
      }}
    }} catch(e) {{}}
    return null;
  }}

  // ---- Text-overlay para colorear @cmd en morado y el resto normal ----
  // Un <input> no puede tener colores mixtos, asi que hacemos el texto del input
  // transparente y ponemos un <div> encima que renderiza las partes con sus colores.
  // NOTA: NO usar innerHTML con font-family entre comillas simples — Python f-string
  // convierte \' en ' y rompe la sintaxis JS. Usamos textContent + style.xxx en su lugar.
  let textOverlay = null;

  function createTextOverlay() {{
    const doc = window.parent.document;
    const el  = doc.createElement('div');
    el.id     = 'chatai_tovl_{input_key}';
    el.style.cssText = 'position:fixed;pointer-events:none;z-index:9998;display:flex;align-items:center;overflow:hidden;white-space:pre;box-sizing:border-box;';
    doc.body.appendChild(el);
    return el;
  }}

  function syncOverlay(inp, val) {{
    if (!textOverlay) textOverlay = createTextOverlay();
    const r  = inp.getBoundingClientRect();
    const cs = window.parent.getComputedStyle(inp);

    // Guardar el color natural del texto (antes de poner transparent) la primera vez
    if (!inp['_chatai_natural_color_{input_key}']) {{
      const nc = cs.color;
      if (nc && nc !== 'transparent' && nc !== 'rgba(0, 0, 0, 0)') {{
        inp['_chatai_natural_color_{input_key}'] = nc;
      }}
    }}
    const naturalColor = inp['_chatai_natural_color_{input_key}'] || '#1f2937';

    textOverlay.style.top           = r.top    + 'px';
    textOverlay.style.left          = r.left   + 'px';
    textOverlay.style.width         = r.width  + 'px';
    textOverlay.style.height        = r.height + 'px';
    textOverlay.style.fontSize      = cs.fontSize;
    textOverlay.style.lineHeight    = cs.lineHeight;
    textOverlay.style.paddingLeft   = cs.paddingLeft;
    textOverlay.style.paddingRight  = cs.paddingRight;
    textOverlay.style.paddingTop    = cs.paddingTop;
    textOverlay.style.paddingBottom = cs.paddingBottom;
    textOverlay.style.boxSizing     = cs.boxSizing;
    textOverlay.style.background    = 'transparent';

    if (val && val.startsWith('@')) {{
      inp.style.color       = 'transparent';
      inp.style.caretColor  = naturalColor;
      inp.style.borderColor = '#7c3aed';
      inp.style.boxShadow   = '0 0 0 2px rgba(124,58,237,.35)';

      const doc      = window.parent.document;
      const spaceIdx = val.indexOf(' ');
      const cmdText  = spaceIdx === -1 ? val : val.slice(0, spaceIdx);
      const restText = spaceIdx === -1 ? ''  : val.slice(spaceIdx);

      textOverlay.innerHTML = '';

      const cmdSpan = doc.createElement('span');
      cmdSpan.textContent      = cmdText;
      cmdSpan.style.color      = '#a78bfa';
      cmdSpan.style.fontFamily = 'JetBrains Mono, monospace';
      cmdSpan.style.fontWeight = '700';
      textOverlay.appendChild(cmdSpan);

      if (restText) {{
        const restSpan = doc.createElement('span');
        restSpan.textContent      = restText;
        restSpan.style.color      = naturalColor;
        restSpan.style.fontFamily = cs.fontFamily;
        restSpan.style.fontWeight = 'normal';
        textOverlay.appendChild(restSpan);
      }}
      textOverlay.style.display = 'flex';
    }} else {{
      inp.style.color       = '';
      inp.style.caretColor  = '';
      inp.style.borderColor = '';
      inp.style.boxShadow   = '';
      textOverlay.style.display = 'none';
    }}
  }}

  function createDropdown() {{
    const doc = window.parent.document;
    const el = doc.createElement('div');
    el.id = 'chatai_dd_{input_key}';
    el.style.cssText = [
      'position:fixed','z-index:999999',
      'background:#0e1117',
      'border:1.5px solid #7c3aed',
      'border-radius:10px',
      'max-height:320px','overflow-y:auto',
      'min-width:360px',
      'box-shadow:0 12px 32px rgba(124,58,237,.3),0 4px 12px rgba(0,0,0,.6)',
      'display:none',
      'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif',
    ].join(';');
    doc.body.appendChild(el);
    return el;
  }}

  function positionDropdown(inp, dd) {{
    const r = inp.getBoundingClientRect();
    dd.style.top   = (r.bottom + 4) + 'px';
    dd.style.left  = r.left + 'px';
    dd.style.width = Math.max(r.width, 360) + 'px';
  }}

  function buildDropdown(filter, inp, dd) {{
    const f = filter.toLowerCase().trim();
    const matches = CMDS.filter(c =>
      !f || c.cmd.slice(1).includes(f) ||
      c.label.toLowerCase().includes(f) ||
      c.desc.toLowerCase().includes(f)
    );
    dd.innerHTML = '';
    if (!matches.length) {{ dd.style.display = 'none'; return; }}

    const cats = {{}};
    matches.forEach(c => {{ (cats[c.cat] = cats[c.cat] || []).push(c); }});

    let first = true;
    Object.entries(cats).forEach(([cat, items]) => {{
      const doc = window.parent.document;
      const catEl = doc.createElement('div');
      catEl.style.cssText =
        'padding:6px 14px 3px;font-size:10px;color:#6b7280;font-weight:700;' +
        'text-transform:uppercase;letter-spacing:.07em;' +
        (first ? '' : 'border-top:1px solid #1f2937;');
      first = false;
      catEl.textContent = cat;
      dd.appendChild(catEl);

      items.forEach(c => {{
        const row = doc.createElement('div');
        row.dataset.value = c.cmd;
        row.style.cssText = 'padding:8px 14px 7px;cursor:pointer;transition:background .1s;';
        row.innerHTML =
          '<div style="display:flex;align-items:center;gap:8px;margin-bottom:2px;">' +
          '<span style="font-size:12px;color:#a78bfa;font-family:monospace;font-weight:700;' +
          'background:#2d1f4a;padding:1px 6px;border-radius:4px;">' + c.cmd + '</span>' +
          '<span style="font-size:13px;color:#e2e8f0;font-weight:500;">' + c.label + '</span>' +
          '</div>' +
          '<div style="font-size:11px;color:#6b7280;">' + c.desc + '</div>';
        row.addEventListener('mouseenter', () => {{
          dd.querySelectorAll('.chatai-active').forEach(x => {{ x.style.background=''; x.classList.remove('chatai-active'); }});
          row.style.background = '#1e1635'; row.classList.add('chatai-active');
        }});
        row.addEventListener('mouseleave', () => {{ row.style.background = ''; row.classList.remove('chatai-active'); }});
        row.addEventListener('mousedown', e => {{ e.preventDefault(); selectCmd(c.cmd, inp, dd); }});
        dd.appendChild(row);
      }});
    }});

    positionDropdown(inp, dd);
    dd.style.display = 'block';
  }}

  function selectCmd(cmd, inp, dd) {{
    // Insertar el comando + espacio para que el usuario pueda agregar instrucciones.
    // NO auto-ejecutar; el usuario presiona Enter o el boton cuando este listo.
    const val = cmd + ' ';
    try {{
      const setter = Object.getOwnPropertyDescriptor(window.parent.HTMLInputElement.prototype, 'value').set;
      setter.call(inp, val);
    }} catch(e) {{ inp.value = val; }}
    inp.dispatchEvent(new window.parent.Event('input',  {{ bubbles: true }}));
    inp.dispatchEvent(new window.parent.Event('change', {{ bubbles: true }}));
    dd.style.display = 'none';
    syncOverlay(inp, val);  // val tiene espacio -> instrucciones en estilo normal
    inp.focus();
    // Mover cursor al final
    try {{ inp.setSelectionRange(val.length, val.length); }} catch(e) {{}}
  }}

  function attachToInput(inp) {{
    if (inp['_chatai_{input_key}']) return;
    inp['_chatai_{input_key}'] = true;

    if (!dropdown) dropdown = createDropdown();
    syncOverlay(inp, inp.value);

    inp.addEventListener('input', () => {{
      const val = inp.value;
      syncOverlay(inp, val);
      if (val.startsWith('@')) buildDropdown(val.slice(1), inp, dropdown);
      else dropdown.style.display = 'none';
    }});

    inp.addEventListener('keydown', e => {{
      const ddVisible = dropdown && dropdown.style.display !== 'none';
      if (ddVisible) {{
        const rows   = [...dropdown.querySelectorAll('[data-value]')];
        const active = dropdown.querySelector('.chatai-active');
        if (e.key === 'ArrowDown') {{
          e.preventDefault();
          const next = rows[(active ? rows.indexOf(active)+1 : 0) % rows.length];
          if (active) {{ active.style.background=''; active.classList.remove('chatai-active'); }}
          next.style.background = '#1e1635'; next.classList.add('chatai-active'); return;
        }}
        if (e.key === 'ArrowUp') {{
          e.preventDefault();
          const idx  = active ? rows.indexOf(active) : rows.length;
          const prev = rows[(idx-1+rows.length) % rows.length];
          if (active) {{ active.style.background=''; active.classList.remove('chatai-active'); }}
          prev.style.background = '#1e1635'; prev.classList.add('chatai-active'); return;
        }}
        if (e.key === 'Enter') {{
          e.preventDefault();
          if (active) selectCmd(active.dataset.value, inp, dropdown);
          else {{ dropdown.style.display='none'; const b=findApplyBtn(); if(b) b.click(); }}
          return;
        }}
        if (e.key === 'Escape') {{ dropdown.style.display='none'; return; }}
      }} else if (e.key === 'Enter' && !e.shiftKey) {{
        e.preventDefault();
        const b = findApplyBtn(); if (b) b.click();
      }}
    }});

    inp.addEventListener('blur', () => {{ setTimeout(() => {{ if (dropdown) dropdown.style.display='none'; }}, 160); }});
    window.parent.document.addEventListener('click', e => {{
      if (dropdown && !dropdown.contains(e.target) && e.target !== inp) dropdown.style.display='none';
    }}, true);
    // Re-sync overlay position on scroll/resize
    const repos = () => {{ if (textOverlay && textOverlay.style.display!=='none') syncOverlay(inp, inp.value); }};
    window.parent.document.addEventListener('scroll', repos, true);
    window.addEventListener('resize', () => {{
      repos();
      if (dropdown && dropdown.style.display!=='none') positionDropdown(inp, dropdown);
    }});
  }}

  // ---- Limpiar divs flotantes de renders anteriores (Streamlit re-renderiza el iframe) ----
  (function cleanup() {{
    const old1 = window.parent.document.getElementById('chatai_tovl_{input_key}');
    if (old1) old1.remove();
    const old2 = window.parent.document.getElementById('chatai_dd_{input_key}');
    if (old2) old2.remove();
  }})();

  // Retry loop \u2014 Streamlit renderiza el input de forma asincrona
  let tries = 0;
  let attachedInp = null;
  const t = setInterval(() => {{
    const inp = findInput();
    if (inp) {{
      attachToInput(inp);
      attachedInp = inp;
      clearInterval(t);

      // Watchdog: limpia el overlay si el input ya no esta en el DOM
      // (Streamlit a veces swap-ea el input sin reload del iframe) + re-sync posicion
      setInterval(() => {{
        // 1) Si el input desaparecio del DOM, esconder overlay + dropdown
        if (!attachedInp || !attachedInp.isConnected) {{
          if (textOverlay) textOverlay.style.display = 'none';
          if (dropdown)    dropdown.style.display    = 'none';
          // Reintentar encontrar el nuevo input
          const freshInp = findInput();
          if (freshInp && freshInp !== attachedInp) {{
            attachedInp = freshInp;
            attachToInput(freshInp);
          }}
          return;
        }}
        // 2) Polling de valor (Streamlit borra programaticamente sin disparar 'input')
        const v = attachedInp.value;
        const overlayShown = textOverlay && textOverlay.style.display !== 'none';
        if (!v.startsWith('@') && overlayShown) syncOverlay(attachedInp, v);
        // 3) Re-posicionar overlay si se movio (scroll interno, layout shift)
        if (overlayShown) {{
          const r = attachedInp.getBoundingClientRect();
          if (Math.abs(parseFloat(textOverlay.style.top)  - r.top)  > 1 ||
              Math.abs(parseFloat(textOverlay.style.left) - r.left) > 1) {{
            syncOverlay(attachedInp, v);
          }}
        }}
      }}, 200);
    }}
    if (++tries > 40) clearInterval(t);
  }}, 200);
}})();
</script></body></html>
"""
    _stcomp.html(html, height=1, scrolling=False)


# ---- LEGACY: kept for reference only, now replaced by _inject_autocomplete_js above ----
def _inject_autocomplete_js_OLD(input_key: str, submit_btn_key: str) -> None:
    """OLD — st.markdown script tags don't execute in React dangerouslySetInnerHTML."""
    pass

    js_code = f"""
<datalist id="chatai_cmds_{input_key}">
</datalist>
<script>
(function() {{
  function findInput() {{
    const all = document.querySelectorAll('input[type="text"]');
    for (const el of all) {{
      if (el.placeholder && el.placeholder.includes('@')) return el;
    }}
    return null;
  }}

  function attachAutocomplete(inp) {{
    if (inp._chatai_attached) return;
    inp._chatai_attached = true;

    inp.addEventListener('input', () => {{
      const val = inp.value;
      if (val.startsWith('@')) {{
        buildItems(val.slice(1));
      }} else {{
        dropdown.style.display = 'none';
      }}
    }});

    inp.addEventListener('keydown', (e) => {{
      if (dropdown.style.display !== 'none') {{
        const active = dropdown.querySelector('.chatai-active');
        const rows   = [...dropdown.querySelectorAll('[data-value]')];
        if (e.key === 'ArrowDown') {{
          e.preventDefault();
          const idx = active ? rows.indexOf(active) : -1;
          const next = rows[(idx + 1) % rows.length];
          if (active) {{ active.style.background=''; active.classList.remove('chatai-active'); }}
          next.style.background = '#2d3748';
          next.classList.add('chatai-active');
          return;
        }}
        if (e.key === 'ArrowUp') {{
          e.preventDefault();
          const idx = active ? rows.indexOf(active) : rows.length;
          const prev = rows[(idx - 1 + rows.length) % rows.length];
          if (active) {{ active.style.background=''; active.classList.remove('chatai-active'); }}
          prev.style.background = '#2d3748';
          prev.classList.add('chatai-active');
          return;
        }}
        if (e.key === 'Enter' && active) {{
          e.preventDefault();
          selectItem(active.dataset.value);
          return;
        }}
        if (e.key === 'Escape') {{
          dropdown.style.display = 'none';
          return;
        }}
      }}
      if (e.key === 'Enter' && !e.shiftKey) {{
        e.preventDefault();
        submitPrompt();
      }}
    }});

    document.addEventListener('click', (e) => {{
      if (!dropdown.contains(e.target) && e.target !== inp) {{
        dropdown.style.display = 'none';
      }}
    }});
  }}

  // Retry hasta encontrar el input (Streamlit renderiza async)
  let attempts = 0;
  const timer = setInterval(() => {{
    const inp = findInput();
    if (inp) {{ attachAutocomplete(inp); clearInterval(timer); }}
    if (++attempts > 40) clearInterval(timer);
  }}, 150);
}})();
</script>
""".strip()
    st.markdown(js_code, unsafe_allow_html=True)


def _run_code_free_mode(instructions: str, state: dict, state_key: str, df, context, summary: dict) -> None:
    """Modo @code: la IA genera Python con libertad total y se ejecuta en el sandbox."""
    # Extraer spec Vega de la grafica actual
    chart_spec: dict = {}
    try:
        _sk    = state_key.replace("chartai_", "")
        _store = st.session_state.get(f"chartai_{_sk}_store", {})
        _c     = _store.get("chart")
        if _c is not None and hasattr(_c, "to_dict"):
            full = _c.to_dict()
            chart_spec = {
                "mark":     full.get("mark"),
                "encoding": full.get("encoding"),
                "transform": full.get("transform"),
                "width":    full.get("width"),
                "height":   full.get("height"),
            }
    except Exception:
        chart_spec = {}

    current_overlays = [
        {k: v for k, v in ov.items() if k != "data"}
        for ov in state.get("overlays", [])
    ]

    # Resumen compacto: quitar recent_data/histogram_bins — el df completo ya esta disponible
    # en el sandbox, no hay necesidad de serializar los puntos crudos (evita 500 por contexto largo)
    compact_summary = {
        k: v for k, v in summary.items()
        if k not in ("recent_data", "histogram_bins", "correlation_pairs",
                     "top_positive_correlations", "top_negative_correlations")
    }
    # Agregar muestra pequeña (10 filas) para que el modelo sepa columnas y tipos
    if df is not None and not df.empty:
        sample_rows = df.head(5).copy()
        for col in sample_rows.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]):
            sample_rows[col] = sample_rows[col].astype(str)
        compact_summary["columns"] = list(df.columns)
        compact_summary["sample_5_rows"] = sample_rows.to_dict(orient="records")

    # Describir la spec de forma textual para no confundir al modelo con un JSON extra
    spec_desc = ""
    if chart_spec:
        mark = chart_spec.get("mark") or {}
        mark_type = mark.get("type", mark) if isinstance(mark, dict) else mark
        enc = chart_spec.get("encoding") or {}
        x_field = (enc.get("x") or {}).get("field", "?")
        y_field = (enc.get("y") or {}).get("field", "?")
        spec_desc = f"\n\nGrafica actual: tipo={mark_type}, eje X={x_field}, eje Y={y_field}"

    user_msg = (
        "Peticion del usuario: " + instructions
        + f"\n\nDatos (resumen estadistico):\n{json.dumps(compact_summary, ensure_ascii=False, indent=2)}"
        + (f"\n\nOverlays activos (labels): {[o.get('label','?') for o in current_overlays]}"
           if current_overlays else "")
        + spec_desc
        + "\n\n---"
        "\nRECORDATORIO CRITICO para este @code:\n"
        "- Para marcar puntos discretos (anomalias, outliers) USA type=\"points\" (NO \"series\").\n"
        "- SIEMPRE que el analisis produzca valores relevantes, agrega tambien una tabla en sub_charts (type=\"table\", df=<pd.DataFrame>).\n"
        "- Rellena result['overlays'] Y result['sub_charts'] Y result['explanation'] — los 3 campos.\n"
        "- Sin emojis ni markdown en explanation; texto llano en espanol.\n"
        "\nResponde UNICAMENTE con:\n"
        '{"mode":"code","python_code":"<codigo Python completo>","overlays":[],"explanation":"<descripcion>"}'
    )

    try:
        with st.spinner("🧠 IA generando codigo personalizado..."):
            # Usamos _CODE_SYSTEM (probado, funciona con force_json=True) forzando mode=code
            result = chat_completion_json(_CODE_SYSTEM, user_msg, temperature=0.2)
    except Exception as e:
        st.error(f"Error contactando al modelo: {e}")
        return

    if "error" in result:
        st.error("⚠️ No pude interpretar la respuesta del modelo.")
        with st.expander("Respuesta cruda (debug)"):
            st.code(result.get("raw", ""), language="text")
        return

    # Aceptar nombres alternativos que el modelo pueda usar
    code = (
        result.get("python_code")
        or result.get("code")
        or result.get("script")
        or ""
    )
    explanation = result.get("explanation", "")

    if not code:
        # Si el modelo respondio mode=overlay en lugar de code, reintentamos con mensaje mas directo
        if result.get("mode") == "overlay" or "overlays" in result:
            st.warning("El modelo devolvio overlays en lugar de codigo. Reintentando...")
            retry_msg = (
                "Tu respuesta anterior uso mode='overlay' pero se requiere mode='code' con python_code. "
                f"Peticion: {instructions}. "
                "Responde con: {\"mode\":\"code\",\"python_code\":\"<codigo Python completo>\","
                "\"overlays\":[],\"explanation\":\"<descripcion>\"}"
            )
            try:
                result = chat_completion_json(_CODE_SYSTEM, retry_msg, temperature=0.1)
                code = result.get("python_code") or result.get("code") or ""
                explanation = result.get("explanation", "")
            except Exception:
                pass
        if not code:
            st.warning("La IA no genero codigo.")
            with st.expander("Respuesta del modelo (debug)"):
                st.json(result)
            return

    with st.spinner("⚙️ Ejecutando codigo generado..."):
        exec_res = _execute_llm_code(code, df, chart_spec)

    # Auto-retry: si el codigo falla, enviar el error al LLM para que lo corrija
    if exec_res.get("error"):
        err_short = exec_res["error"].split("\n")[0][:400]
        with st.spinner("🔄 Codigo fallo, pidiendo correccion a la IA..."):
            fix_msg = (
                f"Tu codigo anterior fallo con este error:\n{err_short}\n\n"
                f"Codigo original:\n```python\n{code}\n```\n\n"
                "Corrige el error y devuelve el JSON de nuevo con el codigo arreglado. "
                "RECUERDA: sklearn no acepta datetime — convierte TimeStamp a "
                "(d['TimeStamp'].astype('int64') // 10**9).values.reshape(-1,1) antes de .fit()/.predict().\n"
                f"Peticion original: {instructions}\n\n"
                'Responde UNICAMENTE con {"mode":"code","python_code":"<codigo corregido>","overlays":[],"explanation":"<desc>"}'
            )
            try:
                fix_result = chat_completion_json(_CODE_SYSTEM, fix_msg, temperature=0.1)
                fixed_code = fix_result.get("python_code") or fix_result.get("code") or ""
                if fixed_code:
                    code = fixed_code
                    explanation = fix_result.get("explanation", explanation)
                    exec_res = _execute_llm_code(code, df, chart_spec)
            except Exception:
                pass

    if exec_res.get("error"):
        st.error("⚠️ El codigo fallo al ejecutarse (incluso despues de reintento):")
        st.code(exec_res["error"], language="text")
        with st.expander("📜 Codigo generado por la IA", expanded=True):
            st.code(code, language="python")
            if exec_res.get("stdout"):
                st.caption("**Salida del codigo:**")
                st.code(exec_res["stdout"], language="text")
        return

    new_overlays = exec_res.get("overlays", []) or []
    new_subs     = exec_res.get("sub_charts", []) or []
    expl         = exec_res.get("explanation") or explanation

    if not new_overlays and not new_subs:
        st.info(expl or "El codigo se ejecuto pero no genero overlays ni sub-graficas.")
        with st.expander("📜 Codigo generado por la IA", expanded=False):
            st.code(code, language="python")
        return

    # Procesar overlays (incluyendo replace_by_label igual que en Ruta C)
    added = replaced = 0
    for ov in new_overlays:
        if ov.get("action") == "replace_by_label":
            lbl = ov.get("label", "").lower()
            match_idx = None
            for i, existing in enumerate(state["overlays"]):
                if existing.get("label", "").lower() == lbl:
                    match_idx = i; break
            if match_idx is None:
                for i, existing in enumerate(state["overlays"]):
                    elbl = existing.get("label", "").lower()
                    if lbl and (lbl in elbl or elbl in lbl):
                        match_idx = i; break
            if match_idx is None and state["overlays"]:
                match_idx = len(state["overlays"]) - 1
            if match_idx is not None:
                state["overlays"][match_idx].update(
                    {k: v for k, v in ov.items() if k != "action"}
                )
                replaced += 1
        else:
            state["overlays"].append(ov)
            added += 1

    if new_subs:
        state["sub_charts"] = state.get("sub_charts", []) + new_subs
    state["last_explanation"] = expl

    parts = []
    if added:    parts.append(f"{added} overlay(s) nuevo(s)")
    if replaced: parts.append(f"{replaced} overlay(s) modificado(s)")
    if new_subs: parts.append(f"{len(new_subs)} sub-grafica(s)")

    # Persistir resumen + codigo para que el fragment los muestre despues del rerun
    state["_last_ai_code"] = {
        "summary": f"✓ @code aplicado: {', '.join(parts)}. {expl}",
        "code":    code,
        "stdout":  exec_res.get("stdout", ""),
    }

    # scope=fragment: solo recarga el bloque del chart, mantiene posicion de scroll
    st.rerun(scope="fragment")


def _render_modify_panel(state: dict, state_key: str, df, context):
    prompt_key  = f"{state_key}_prompt_val"
    submit_key  = f"{state_key}_apply_btn"

    # Pre-llenar si viene de click directo en sugerencia (ejecucion inmediata)
    if "_pending_prompt" in state:
        st.session_state[prompt_key] = state.pop("_pending_prompt")

    def _on_change():
        pass  # solo para registrar cambios; la logica de show_cmds la maneja JS

    prompt = st.text_input(
        "IA",
        key=prompt_key,
        on_change=_on_change,
        placeholder="Escribe @ para ver comandos, o describe lo que quieres (Enter para aplicar)",
        label_visibility="collapsed",
    )

    # Boton "Aplicar" invisible (JS lo hace clic al presionar Enter)
    go = st.button(
        "🚀 Aplicar",
        key=submit_key,
        type="primary",
        use_container_width=True,
    )

    # Inyectar el autocomplete JS (se adjunta al input de arriba)
    _inject_autocomplete_js(prompt_key, submit_key)

    if not (go and prompt.strip()):
        return

    # Limpiar input
    if prompt_key in st.session_state:
        del st.session_state[prompt_key]

    summary = _build_summary(df, context)
    p_lower  = prompt.lower()

    # ---- Ruta @code (MAXIMA PRIORIDAD): libertad total a la IA, codigo Python ----
    p_stripped = prompt.strip()
    if p_stripped.lower().startswith("@code"):
        instructions = p_stripped[5:].strip()
        if not instructions:
            st.warning("Escribe instrucciones despues de @code. Ej: `@code detecta anomalias con Isolation Forest`")
            return
        _run_code_free_mode(instructions, state, state_key, df, context, summary)
        return

    # ---- Ruta @ (comandos analiticos predefinidos) ----
    analysis_result = _dispatch_analysis_command(prompt, df, context)
    if analysis_result:
        if "error" in analysis_result:
            st.error(analysis_result["error"])
            if not _HAS_STATSMODELS and "statsmodels" in analysis_result.get("error", ""):
                st.code("pip install statsmodels", language="bash")
            return
        new_overlays = analysis_result.get("overlays", [])
        new_sub      = analysis_result.get("sub_charts", [])
        explanation  = analysis_result.get("explanation", "")
        if new_overlays:
            state["overlays"].extend(new_overlays)
        if new_sub:
            state["sub_charts"] = state.get("sub_charts", []) + new_sub
        state["last_explanation"] = explanation
        if explanation:
            st.success("✓ Análisis completado.")
            with st.expander("📊 Detalle", expanded=False):
                st.markdown(explanation)
        st.rerun()
        return

    has_ts_data = (
        df is not None and not df.empty
        and "Value_Num" in df.columns
        and "TimeStamp" in df.columns
    )

    # ---- Ruta A: Prediccion compleja con LLM (elige modelo) ----
    is_complex_predict = has_ts_data and (
        any(w in p_lower for w in _COMPLEX_PREDICT_WORDS)
        or (any(w in p_lower for w in _SIMPLE_PREDICT_WORDS) and len(df) >= 20)
    )
    if is_complex_predict:
        with st.spinner("🧠 Analizando patron de datos y calculando modelo predictivo..."):
            overlays, explanation, reasoning = _llm_guided_forecast(prompt, df, context, summary)
        if overlays:
            state["overlays"].extend(overlays)
            state["last_explanation"] = explanation
            st.success("✓ Modelo predictivo aplicado.")
            with st.expander("📊 Detalle del modelo", expanded=False):
                st.markdown(f"**{explanation}**")
                if reasoning:
                    st.markdown(f"*{reasoning}*")
        else:
            st.warning("No se pudo calcular el modelo. La grafica necesita datos numericos con timestamps.")
        st.rerun()
        return

    # ---- Ruta B: Fallback local (rapido, sin LLM) ----
    local = _local_overlay_fallback(prompt, summary, df)
    if local:
        state["overlays"].extend(local)
        state["last_explanation"] = f"{len(local)} capa(s) de control aplicada(s)."
        st.success(f"✓ {len(local)} modificacion(es) aplicada(s).")
        st.rerun()
        return

    # ---- Ruta C: LLM libre con acceso total (overlays declarativos O codigo Python) ----
    # Incluir overlays actuales (sin el array 'data' masivo) para que la IA pueda modificarlos
    current_overlays_summary = [
        {k: v for k, v in ov.items() if k != "data"}
        for ov in state.get("overlays", [])
    ]
    # Extraer spec Vega de la grafica para que la IA "vea" su codigo
    chart_spec: dict = {}
    try:
        store = st.session_state.get(f"{state_key[len('chartai_'):]}") if False else None
        # Recuperamos el chart desde el store usando el state_key
        _sk = state_key.replace("chartai_", "")
        _store = st.session_state.get(f"chartai_{_sk}_store", {})
        _c = _store.get("chart")
        if _c is not None and hasattr(_c, "to_dict"):
            full_spec = _c.to_dict()
            # Reducir: quitar datasets grandes inline
            chart_spec = {
                "mark": full_spec.get("mark"),
                "encoding": full_spec.get("encoding"),
                "transform": full_spec.get("transform"),
                "config": {k: v for k, v in (full_spec.get("config") or {}).items() if k in ("view","axis","title")},
                "width": full_spec.get("width"),
                "height": full_spec.get("height"),
            }
    except Exception:
        chart_spec = {}

    user_msg = (
        f"Peticion del usuario: {prompt}\n\n"
        f"Datos de la grafica (resumen):\n{json.dumps(summary, ensure_ascii=False, indent=2)}"
        + (
            f"\n\nOverlays actualmente visibles:\n"
            f"{json.dumps(current_overlays_summary, ensure_ascii=False, indent=2)}"
            if current_overlays_summary else ""
        )
        + (
            f"\n\nSpec Vega-Lite de la grafica actual:\n"
            f"{json.dumps(chart_spec, ensure_ascii=False, indent=2, default=str)[:2000]}"
            if chart_spec else ""
        )
    )
    try:
        with st.spinner("🧠 La IA esta analizando la grafica y tus datos..."):
            result = chat_completion_json(_CODE_SYSTEM, user_msg, temperature=0.1)

        if "error" in result:
            st.error("⚠️ No pude interpretar la respuesta del modelo.")
            with st.expander("Ver respuesta cruda del modelo (debug)"):
                st.code(result.get("raw", ""), language="text")
            st.info("💡 Escribe @ para ver comandos analiticos predefinidos.")
            return

        mode        = (result.get("mode") or "overlay").lower()
        explanation = result.get("explanation", "")

        # ---- Modo CODE: ejecutar Python generado por la IA ----
        if mode == "code" and result.get("python_code"):
            code = result["python_code"]
            with st.spinner("⚙️ Ejecutando analisis..."):
                exec_res = _execute_llm_code(code, df, chart_spec)

            if exec_res.get("error"):
                st.error("⚠️ El codigo generado fallo al ejecutarse.")
                with st.expander("Ver error y codigo (debug)"):
                    st.code(exec_res["error"], language="text")
                    st.code(code, language="python")
                return

            new_overlays = exec_res.get("overlays", []) or []
            new_subs     = exec_res.get("sub_charts", []) or []
            expl_code    = exec_res.get("explanation") or explanation

            if not new_overlays and not new_subs:
                st.warning(expl_code or "El codigo se ejecuto pero no produjo resultados.")
                if exec_res.get("stdout"):
                    with st.expander("Salida del codigo"):
                        st.code(exec_res["stdout"], language="text")
                return

            if new_overlays:
                state["overlays"].extend(new_overlays)
            if new_subs:
                state["sub_charts"] = state.get("sub_charts", []) + new_subs
            state["last_explanation"] = expl_code
            msg_parts = []
            if new_overlays: msg_parts.append(f"{len(new_overlays)} overlay(s)")
            if new_subs:     msg_parts.append(f"{len(new_subs)} sub-grafica(s)")
            st.success(f"✓ IA ejecuto codigo: {', '.join(msg_parts)}. {expl_code}")
            with st.expander("📜 Ver codigo generado por la IA", expanded=False):
                st.code(code, language="python")
            st.rerun()
            return

        # ---- Modo OVERLAY: JSON declarativo (rapido, cambios simples) ----
        new_overlays = result.get("overlays", []) or []
        if not new_overlays:
            st.warning(explanation or "La IA no propuso modificaciones.")
            return

        added = 0
        replaced = 0
        for ov in new_overlays:
            if ov.get("action") == "replace_by_label":
                lbl = ov.get("label", "").lower()
                match_idx = None
                for i, existing in enumerate(state["overlays"]):
                    if existing.get("label", "").lower() == lbl:
                        match_idx = i; break
                if match_idx is None:
                    for i, existing in enumerate(state["overlays"]):
                        elbl = existing.get("label", "").lower()
                        if lbl and (lbl in elbl or elbl in lbl):
                            match_idx = i; break
                if match_idx is None and state["overlays"]:
                    match_idx = len(state["overlays"]) - 1
                if match_idx is not None:
                    state["overlays"][match_idx].update(
                        {k: v for k, v in ov.items() if k != "action"}
                    )
                    replaced += 1
            else:
                state["overlays"].append(ov)
                added += 1
        state["last_explanation"] = explanation
        parts = []
        if replaced: parts.append(f"{replaced} overlay(s) modificado(s)")
        if added:    parts.append(f"{added} overlay(s) agregado(s)")
        st.success(f"✓ {', '.join(parts) or 'cambios aplicados'}. {explanation}")
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
            st.rerun()   # full-page: garantiza que el historial se muestre completo
        except Exception as e:
            st.error(f"Error: {e}")

