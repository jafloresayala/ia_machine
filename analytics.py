"""Helpers analíticos avanzados: estabilidad, outliers, correlación, salud."""
import numpy as np
import pandas as pd


def numeric_summary(df: pd.DataFrame) -> dict:
    """Estadísticas básicas y de estabilidad para un tag numérico."""
    if df.empty or "Value_Num" not in df.columns:
        return {}
    v = df["Value_Num"].dropna()
    if v.empty:
        return {}

    mean = float(v.mean())
    std = float(v.std(ddof=0))
    cv = float(std / mean * 100) if mean not in (0.0, 0) else float("nan")

    q1, q3 = np.percentile(v, [25, 75])
    iqr = float(q3 - q1)

    out_count = int(((v < mean - 2 * std) | (v > mean + 2 * std)).sum())
    out_pct = float(out_count / len(v) * 100)

    # Estabilidad (0-100): menor CV + menos outliers = más estable
    stability = max(0.0, 100.0 - min(abs(cv), 60.0) - out_pct * 2)

    # Health score heurístico
    if std == 0 or np.isnan(std):
        health = 70.0  # plano podría ser un sensor muerto
    else:
        health = max(0.0, min(100.0, stability))

    return {
        "count": int(len(v)),
        "last": float(v.iloc[-1]),
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": mean,
        "median": float(v.median()),
        "std": std,
        "cv_pct": cv,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": iqr,
        "range": float(v.max() - v.min()),
        "outliers_2std": out_count,
        "outlier_pct": out_pct,
        "stability": stability,
        "health": health,
    }


def categorical_summary(df: pd.DataFrame) -> dict:
    """Resumen para tags categóricos / booleanos."""
    if df.empty or "Value_Str" not in df.columns:
        return {}
    v = df["Value_Str"].dropna().astype(str).str.strip()
    v = v[~v.isin(["", "nan", "None", "NaN"])]
    if v.empty:
        return {}
    counts = v.value_counts()
    total = int(counts.sum())
    pcts = (counts / total * 100).round(1).to_dict()
    return {
        "count": total,
        "last": str(v.iloc[-1]),
        "unique_states": int(v.nunique()),
        "distribution": pcts,
        "most_common": counts.index[0],
        "most_common_pct": float(counts.iloc[0] / total * 100),
    }


def correlation_matrix(tag_data: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """Correlación de Pearson entre tags numéricos.

    Alinea por timestamp usando merge_asof (tolerancia flexible).
    """
    numeric_series = {}
    for tag_name, df in tag_data.items():
        if df.empty or "Value_Num" not in df.columns:
            continue
        sub = df[["TimeStamp", "Value_Num"]].dropna()
        if sub.empty or sub["Value_Num"].notna().sum() < 3:
            continue
        sub = sub.sort_values("TimeStamp")
        sub = sub.set_index("TimeStamp")["Value_Num"]
        numeric_series[tag_name] = sub

    if len(numeric_series) < 2:
        return None

    # Resample cada uno a 1 min promedio para alinear
    resampled = {}
    for name, s in numeric_series.items():
        try:
            resampled[name] = s.resample("1min").mean()
        except Exception:
            resampled[name] = s

    frame = pd.DataFrame(resampled)
    if frame.dropna(how="all").empty:
        return None

    return frame.corr(method="pearson", min_periods=3)


def detect_anomalies(df: pd.DataFrame, z_threshold: float = 2.5) -> pd.DataFrame:
    """Detecta puntos anómalos usando z-score sobre Value_Num."""
    if df.empty or "Value_Num" not in df.columns:
        return pd.DataFrame()
    v = df["Value_Num"]
    mean = v.mean()
    std = v.std(ddof=0)
    if not std or np.isnan(std):
        return pd.DataFrame()
    z = (v - mean) / std
    mask = z.abs() > z_threshold
    flagged = df[mask].copy()
    if flagged.empty:
        return flagged
    flagged["z_score"] = z[mask].values
    flagged = flagged.sort_values("z_score", key=lambda s: s.abs(), ascending=False)
    return flagged


def health_score_for_tags(tag_data: dict[str, pd.DataFrame]) -> dict:
    """Calcula un score compuesto de salud para un conjunto de tags."""
    scores = []
    for name, df in tag_data.items():
        s = numeric_summary(df)
        if s:
            scores.append(s["health"])
    if not scores:
        return {"health": None, "numeric_tags": 0}
    return {
        "health": float(np.mean(scores)),
        "numeric_tags": len(scores),
        "min_health": float(np.min(scores)),
    }
