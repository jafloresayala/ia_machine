"""
Registro / catálogo de máquinas.

Descubre la jerarquía PI (Línea → Máquina) al iniciar y la cachea.
Permite buscar máquinas por nombre y obtener sus atributos / tags.
"""
import json
import os
import streamlit as st
import pandas as pd
from datetime import datetime
from api_client import fetch_children, fetch_attributes
from config import ROOT_PATH

# Caché en disco — persiste entre reinicios de Streamlit
_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_machines_cache.json")


def _load_disk_cache() -> tuple[pd.DataFrame, datetime | None]:
    """Lee el catálogo desde disco. Devuelve (df_vacío, None) si no existe."""
    if not os.path.exists(_CACHE_FILE):
        return pd.DataFrame(), None
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data["rows"])
        ts = datetime.fromisoformat(data["saved_at"])
        return df, ts
    except Exception:
        return pd.DataFrame(), None


def _save_disk_cache(df: pd.DataFrame) -> None:
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"rows": df.to_dict("records"), "saved_at": datetime.now().isoformat()},
                f, ensure_ascii=False,
            )
    except Exception:
        pass


def get_catalog_info() -> dict:
    """Devuelve info del catálogo en disco: {count, saved_at, has_cache}."""
    df, ts = _load_disk_cache()
    return {"count": len(df), "saved_at": ts, "has_cache": not df.empty}


@st.cache_data(show_spinner="Descubriendo máquinas…")
def _fetch_machines_from_api() -> pd.DataFrame:
    """Recorrido completo de la jerarquía PI. Sin TTL — controlado manualmente."""
    rows = []
    lines_df = fetch_children(ROOT_PATH)

    if lines_df.empty:
        return pd.DataFrame(columns=["line_name", "machine_name", "machine_path"])

    for _, line in lines_df.iterrows():
        line_name = str(line["name"])
        line_path = str(line["path"])

        try:
            machines_df = fetch_children(line_path)
        except Exception:
            continue

        if machines_df.empty:
            continue

        for _, machine in machines_df.iterrows():
            rows.append({
                "line_name": line_name,
                "machine_name": str(machine["name"]),
                "machine_path": str(machine["path"]),
            })

    return pd.DataFrame(rows)


def discover_machines(force_refresh: bool = False) -> pd.DataFrame:
    """
    Devuelve el catálogo de máquinas.
    - Sin force_refresh: usa caché en disco si está disponible (sobrevive reinicios).
    - force_refresh=True: limpia la caché en memoria y en disco, vuelve a escanear PI.
    """
    if not force_refresh:
        cached_df, _ = _load_disk_cache()
        if not cached_df.empty:
            return cached_df

    # Limpiar caché en memoria de Streamlit para forzar re-fetch
    _fetch_machines_from_api.clear()
    df = _fetch_machines_from_api()
    if not df.empty:
        _save_disk_cache(df)
    return df


def get_machine_names() -> list[str]:
    df = discover_machines()
    if df.empty:
        return []
    return sorted(df["machine_name"].unique().tolist())


def get_lines() -> list[str]:
    df = discover_machines()
    if df.empty:
        return []
    return sorted(df["line_name"].unique().tolist())


def find_machine(query: str, line_hint: str | None = None) -> pd.DataFrame:
    """
    Busca máquinas cuyo nombre contenga el texto dado (case-insensitive).
    Si se proporciona line_hint, filtra también por línea para desambiguar.
    Devuelve las filas coincidentes del catálogo.
    """
    df = discover_machines()
    if df.empty:
        return df
    mask = df["machine_name"].str.lower().str.contains(query.lower(), na=False)
    results = df[mask].reset_index(drop=True)

    # Si hay hint de línea y hay múltiples resultados, intentar filtrar
    if line_hint and len(results) > 1:
        hint_lower = line_hint.lower().strip()
        # Intentar match en machine_name (ej: "L1L" en "B2_L1L Paste Printer")
        name_mask = results["machine_name"].str.lower().str.contains(hint_lower, na=False)
        # Intentar match en line_name (ej: "linea 1 left" en "Linea 1 Left")
        line_mask = results["line_name"].str.lower().str.contains(hint_lower, na=False)
        combined = name_mask | line_mask
        if combined.any():
            results = results[combined].reset_index(drop=True)

    return results


@st.cache_data(ttl=600, show_spinner="Cargando atributos…")
def get_machine_attributes(machine_path: str) -> pd.DataFrame:
    return fetch_attributes(machine_path)


def get_machine_tags(machine_path: str) -> pd.DataFrame:
    """
    Devuelve solo los atributos con piPoint válido (= tags consultables).
    """
    attrs = get_machine_attributes(machine_path)
    if attrs.empty or "piPoint" not in attrs.columns:
        return pd.DataFrame(columns=["name", "piPoint", "type", "UOM"])

    valid = attrs[attrs["piPoint"].notna()].copy()
    valid = valid[valid["piPoint"].astype(str).str.strip() != ""]
    keep = ["name", "piPoint", "type", "UOM"]
    for c in keep:
        if c not in valid.columns:
            valid[c] = None
    return valid[keep].drop_duplicates().reset_index(drop=True)


# Palabras clave de tipo de maquina -> terminos de busqueda
_MACHINE_TYPE_KEYWORDS = {
    "reflow":       ["reflow", "horno", "oven"],
    "paste printer":["paste printer", "printer", "impresora", "solder paste"],
    "pick and place":["pick and place", "pick&place", "p&p", "mounters", "mounter", "colocadora"],
    "spi":          ["spi", "solder paste inspection"],
    "aoi":          ["aoi", "optical inspection", "inspeccion optica"],
    "conveyor":     ["conveyor", "cinta", "transportador"],
    "buffer":       ["buffer", "cooling"],
    "loader":       ["loader", "cargador"],
    "unloader":     ["unloader", "descargador"],
}


def smart_find_machine(query: str, line_hint: str | None = None) -> pd.DataFrame:
    """
    Busqueda inteligente de maquinas:
    1. Intenta find_machine exacto con el query
    2. Si no hay resultado, descompone el query en tokens y busca por palabra clave de tipo
    3. Aplica line_hint si se proporciona
    Retorna las filas coincidentes, o un DataFrame vacio si nada encaja.
    """
    # Paso 1: busqueda directa
    result = find_machine(query, line_hint=line_hint) if query else pd.DataFrame()
    if not result.empty:
        return result

    df = discover_machines()
    if df.empty:
        return df

    q_lower = (query or "").lower()

    # Paso 2: buscar por tipo de maquina usando sinonimos
    for machine_type, synonyms in _MACHINE_TYPE_KEYWORDS.items():
        if any(syn in q_lower for syn in synonyms):
            # buscar ese tipo en los nombres de maquina
            type_mask = df["machine_name"].str.lower().str.contains(
                "|".join(synonyms), na=False, regex=True
            )
            candidates = df[type_mask].reset_index(drop=True)
            if not candidates.empty:
                # Aplicar line_hint si existe
                if line_hint:
                    hint_lower = line_hint.lower().strip()
                    name_mask = candidates["machine_name"].str.lower().str.contains(hint_lower, na=False)
                    line_mask = candidates["line_name"].str.lower().str.contains(hint_lower, na=False)
                    filtered = candidates[name_mask | line_mask]
                    if not filtered.empty:
                        return filtered.reset_index(drop=True)
                return candidates

    # Paso 3: buscar token a token (cada palabra del query)
    tokens = [t for t in q_lower.split() if len(t) >= 3]
    for token in tokens:
        mask = df["machine_name"].str.lower().str.contains(token, na=False)
        candidates = df[mask].reset_index(drop=True)
        if not candidates.empty:
            if line_hint:
                hint_lower = line_hint.lower().strip()
                name_mask = candidates["machine_name"].str.lower().str.contains(hint_lower, na=False)
                line_mask = candidates["line_name"].str.lower().str.contains(hint_lower, na=False)
                filtered = candidates[name_mask | line_mask]
                if not filtered.empty:
                    return filtered.reset_index(drop=True)
            return candidates

    return pd.DataFrame()


def build_machine_context_for_llm() -> str:
    """
    Genera un texto compacto con la lista de máquinas/líneas
    para inyectar como contexto al LLM.
    """
    df = discover_machines()
    if df.empty:
        return "No se han descubierto máquinas aún."

    lines = []
    for line_name, group in df.groupby("line_name"):
        machines = ", ".join(group["machine_name"].tolist())
        lines.append(f"- Línea '{line_name}': {machines}")

    return "Máquinas disponibles:\n" + "\n".join(lines)
