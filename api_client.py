"""
Cliente limpio para PI Web API.
Encapsula todas las llamadas HTTP y normalización de datos.
"""
import urllib3
import requests
import pandas as pd
import numpy as np
from datetime import datetime

from config import (
    ENDPOINT_CHILDREN,
    ENDPOINT_ATTRIBUTES,
    ENDPOINT_TAG_VALUES,
    PLUGIN_NAME,
    VERIFY_SSL,
    API_TIMEOUT,
)


def _post(url: str, payload: dict) -> dict | list:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if not VERIFY_SSL:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    resp = requests.post(
        url, headers=headers, json=payload,
        timeout=API_TIMEOUT, verify=VERIFY_SSL,
    )
    resp.raise_for_status()
    return resp.json()


# ----------------------------------------------------------
# Hijos de un elemento
# ----------------------------------------------------------
def fetch_children(element_path: str) -> pd.DataFrame:
    data = _post(ENDPOINT_CHILDREN, {
        "Plugin_Name": PLUGIN_NAME,
        "Element_Path": element_path,
    })

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return pd.DataFrame()

    df = pd.DataFrame(data)
    for col in ("id", "name", "path", "type", "template", "description"):
        if col not in df.columns:
            df[col] = None
    return df[["id", "name", "path", "type", "template", "description"]]


# ----------------------------------------------------------
# Atributos de un elemento
# ----------------------------------------------------------
def fetch_attributes(element_path: str) -> pd.DataFrame:
    data = _post(ENDPOINT_ATTRIBUTES, {
        "Plugin_Name": PLUGIN_NAME,
        "Element_Path": element_path,
    })

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return pd.DataFrame()

    df = pd.DataFrame(data)
    expected = [
        "id", "name", "path", "type", "description", "value",
        "lastValueDate", "configuration_string", "categories",
        "UOM", "hasChildren", "isExcluded", "piPoint",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = None
    return df[expected]


# ----------------------------------------------------------
# Valores de tags en un rango
# ----------------------------------------------------------
def _format_dt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def fetch_tag_values(
    tag_names: list[str],
    from_dt: datetime,
    to_dt: datetime,
) -> pd.DataFrame:
    payload = {
        "Plugin_Name": PLUGIN_NAME,
        "From_Date": _format_dt(from_dt),
        "To_Date": _format_dt(to_dt),
        "Tag_Names": tag_names,
    }
    data = _post(ENDPOINT_TAG_VALUES, payload)

    records = []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return pd.DataFrame()

    for tag_obj in data:
        tag_name = tag_obj.get("Tag_Name")
        tag_type = tag_obj.get("Tag_Type")
        result = tag_obj.get("Result")
        error_msg = tag_obj.get("ErrorMsg")
        for point in tag_obj.get("Tag_Values", []) or []:
            raw = point.get("Value")
            records.append({
                "Tag_Name": tag_name,
                "Tag_Type": tag_type,
                "Result": result,
                "ErrorMsg": error_msg,
                "Value_Raw": raw,
                "Value_Str": "" if raw is None else str(raw),
                "TimeStamp": point.get("TimeStamp"),
            })

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=[
            "Tag_Name", "Tag_Type", "Result", "ErrorMsg",
            "Value_Raw", "Value_Str", "TimeStamp", "Value_Num",
        ])

    # PI devuelve timestamps con offset local embebido (ej: "2026-04-15T11:00:00-05:00").
    # Parsear SIN utc=True preserva ese horario local; luego quitamos el tz-info
    # para dejar un datetime naive con la hora correcta de planta.
    raw_ts = pd.to_datetime(df["TimeStamp"], errors="coerce")
    if pd.api.types.is_datetime64_any_dtype(raw_ts) and raw_ts.dt.tz is not None:
        # Hay offset embebido: preservar la hora de reloj, quitar info de zona
        df["TimeStamp"] = raw_ts.apply(
            lambda x: x.replace(tzinfo=None) if x is not pd.NaT else pd.NaT
        )
    else:
        df["TimeStamp"] = raw_ts
    df["Value_Num"] = pd.to_numeric(df["Value_Raw"], errors="coerce")
    df = df.sort_values("TimeStamp", na_position="last").reset_index(drop=True)
    return df
