"""Lógica de detección y emulación de data drift (features y variable objetivo).

Lógica desacoplada de MLflow y Prefect para facilitar su comprobación
con pruebas unitarias sin dependencias externas de servicios.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
except (ImportError, ModuleNotFoundError):
    try:
        from evidently.legacy.report import Report  # type: ignore
        from evidently.legacy.metric_preset import DataDriftPreset  # type: ignore
    except (ImportError, ModuleNotFoundError):
        Report = None  # type: ignore
        DataDriftPreset = None  # type: ignore


@dataclass(frozen=True)
class ResultadoDriftFeatures:
    total_features: int
    features_con_drift: int
    fraccion_drift: float
    umbral_fraccion: float
    hay_alerta: bool
    columnas_con_drift: list[str]
    detalles_columnas: dict[str, Any]
    mensaje: str


@dataclass(frozen=True)
class ResultadoDriftTarget:
    variable_objetivo: str
    media_referencia: float
    media_lote: float
    diferencia_media_pct: float
    estadistico_ks: float
    p_valor: float
    umbral_p_valor: float
    hay_drift: bool
    hay_alerta: bool
    mensaje: str


def dividir_referencia_y_prueba(
    df: pd.DataFrame,
    test_size: float = 0.2,
    semilla: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el dataset en datos de referencia (entrenamiento) y datos de prueba.

    Garantiza la misma partición que la etapa de entrenamiento.
    """
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=semilla)
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def tomar_lote_prueba(
    df_test: pd.DataFrame,
    tamano_lote: int = 100,
    semilla: int | None = None,
) -> pd.DataFrame:
    """Extrae un lote representativo de la partición de prueba."""
    if len(df_test) <= tamano_lote:
        return df_test.copy()
    if semilla is not None:
        return df_test.sample(n=tamano_lote, random_state=semilla).reset_index(drop=True)
    return df_test.sample(n=tamano_lote).reset_index(drop=True)


def aplicar_drift_emulado(
    lote: pd.DataFrame,
    nivel: float = 0.0,
    objetivo: str = "precio_venta",
    semilla: int = 42,
) -> pd.DataFrame:
    """Aplica drift sintético/emulado controlado por un nivel de 0.0 a 1.0.

    Estrategia de emulación de drift:
    --------------------------------
    1. Nivel 0.0: No realiza ninguna alteración (distribución idéntica al conjunto de prueba).
    2. Nivel > 0.0:
       - 'kilometraje' (numérica): se incrementa en proporción directa al nivel
         (+ nivel * 100.000 km), simulando que en el lote actual ingresaron vehículos con
         mucho mayor desgaste.
       - 'año_modelo' (numérica): se reduce la antigüedad en hasta 4 años según el nivel
         (- round(nivel * 4)), simulando vehículos más viejos.
       - 'precio_lista' (numérica, si existe): se desplaza en proporción (- nivel * 30 %).
       - 'tipo_vehiculo' (categórica, si existe): se sesga la proporción reemplazando
         aleatoriamente una fracción de vehículos por "Pick-up".
       - Variable objetivo ('precio_venta'): se altera su distribución aplicando un
         descuento/caída promedio proporcional al nivel: precio * (1.0 - nivel * 0.35).
         Esto emula el 'event drift' o cambio en el comportamiento del mercado.
    """
    if nivel < 0.0 or nivel > 1.0:
        raise ValueError(f"El nivel de drift debe estar entre 0.0 y 1.0, recibido: {nivel}")

    if nivel == 0.0:
        return lote.copy()

    df_drift = lote.copy()
    rng = np.random.default_rng(semilla)

    # 1. Desplazamiento de kilometraje
    if "kilometraje" in df_drift.columns:
        incremento_km = float(nivel * 100_000.0)
        ruido = rng.normal(0, max(1.0, 4000.0 * nivel), size=len(df_drift))
        df_drift["kilometraje"] = (df_drift["kilometraje"] + incremento_km + ruido).clip(lower=0).round().astype(int)

    # 2. Desplazamiento de año_modelo
    if "año_modelo" in df_drift.columns:
        delta_ano = int(round(nivel * 4))
        df_drift["año_modelo"] = (df_drift["año_modelo"] - delta_ano).clip(lower=2010, upper=2030).astype(int)

    # 3. Desplazamiento de precio_lista
    if "precio_lista" in df_drift.columns:
        factor_lista = max(0.1, 1.0 - (nivel * 0.30))
        df_drift["precio_lista"] = (df_drift["precio_lista"] * factor_lista).round(2)

    # 4. Sesgo de categoría en tipo_vehiculo
    if "tipo_vehiculo" in df_drift.columns:
        fraccion_cambio = min(0.9, nivel * 0.8)
        mascara = rng.random(size=len(df_drift)) < fraccion_cambio
        df_drift.loc[mascara, "tipo_vehiculo"] = "Pick-up"

    # 5. Desplazamiento de la variable objetivo (event drift)
    if objetivo in df_drift.columns:
        descuento_target = max(0.1, 1.0 - (nivel * 0.35))
        ruido_target = rng.normal(1.0, 0.02 * nivel, size=len(df_drift))
        df_drift[objetivo] = (df_drift[objetivo] * descuento_target * ruido_target).clip(lower=500.0).round(2)

    return df_drift


def detectar_drift_features(
    datos_referencia: pd.DataFrame,
    lote: pd.DataFrame,
    columnas_features: list[str] | None = None,
    umbral_fraccion: float = 0.3,
    umbral_p_valor: float = 0.05,
) -> tuple[ResultadoDriftFeatures, Any]:
    """Evalúa el drift de las variables de entrada usando Evidently.

    Retorna una tupla (ResultadoDriftFeatures, reporte_evidently).
    """
    if Report is None or DataDriftPreset is None:
        raise RuntimeError("Evidently no está instalado o no se pudo importar.")

    if columnas_features is None:
        # Excluir fechas y objetivos comunes
        columnas_features = [
            c for c in datos_referencia.columns
            if c in lote.columns and c not in ["fecha_venta", "precio_venta", "precio_predicho", "error_absoluto"]
        ]

    ref_subset = datos_referencia[columnas_features].copy()
    cur_subset = lote[columnas_features].copy()

    # Ejecutar reporte de Evidently
    reporte = Report([DataDriftPreset(drift_share=umbral_fraccion)])
    reporte_run = reporte.run(reference_data=ref_subset, current_data=cur_subset)

    # En Evidently 0.7 reporte.run() devuelve un Run object con .dict()
    datos_dict = reporte_run.dict() if hasattr(reporte_run, "dict") else reporte.as_dict()

    columnas_con_drift: list[str] = []
    detalles_columnas: dict[str, Any] = {}
    fraccion_drift = 0.0
    features_con_drift = 0
    total_features = len(columnas_features)

    for item in datos_dict.get("metrics", []):
        metric_name = str(item.get("metric_name", ""))
        val = item.get("value")

        if "DriftedColumnsCount" in metric_name and isinstance(val, dict):
            features_con_drift = int(val.get("count", 0))
            fraccion_drift = float(val.get("share", 0.0))
        elif "ValueDrift" in metric_name:
            # Ej: ValueDrift(column=kilometraje,method=...,threshold=0.05)
            # Extraer nombre de columna del string o config
            cfg = item.get("config", {})
            col = cfg.get("column")
            if not col and "column=" in metric_name:
                try:
                    col = metric_name.split("column=")[1].split(",")[0]
                except Exception:
                    col = metric_name

            # En Evidently 0.7 el valor puede ser p-valor o distancia
            drift_score = float(val) if isinstance(val, (int, float)) else 0.0
            # Si el método es p-value, drift si score < umbral. Si es distancia, drift si score > umbral
            metodo = str(cfg.get("method", metric_name))
            es_drift = (drift_score < umbral_p_valor) if ("p_value" in metodo.lower() or "test" in metodo.lower()) else (drift_score > 0.1)

            detalles_columnas[col] = {
                "score": drift_score,
                "metodo": metodo,
                "drift_detectado": es_drift,
            }
            if es_drift and col not in columnas_con_drift:
                columnas_con_drift.append(col)

    # Si DriftedColumnsCount no apareció directamente, calcular a partir de las columnas analizadas
    if total_features > 0 and fraccion_drift == 0.0 and len(columnas_con_drift) > 0:
        features_con_drift = len(columnas_con_drift)
        fraccion_drift = round(features_con_drift / total_features, 4)

    hay_alerta = fraccion_drift >= umbral_fraccion
    mensaje = (
        f"Drift en features: {features_con_drift}/{total_features} variables ({fraccion_drift:.1%}). "
        f"Umbral de alerta: {umbral_fraccion:.1%}. "
        f"Alerta: {'ACTIVADA' if hay_alerta else 'NORMAL'}. "
        f"Variables con drift: {columnas_con_drift or 'ninguna'}."
    )

    resultado = ResultadoDriftFeatures(
        total_features=total_features,
        features_con_drift=features_con_drift,
        fraccion_drift=fraccion_drift,
        umbral_fraccion=umbral_fraccion,
        hay_alerta=hay_alerta,
        columnas_con_drift=columnas_con_drift,
        detalles_columnas=detalles_columnas,
        mensaje=mensaje,
    )
    return resultado, reporte_run


def detectar_drift_target(
    datos_referencia: pd.DataFrame,
    lote: pd.DataFrame,
    objetivo: str = "precio_venta",
    umbral_p_valor: float = 0.05,
) -> ResultadoDriftTarget:
    """Evalúa el drift de la variable objetivo (event drift) frente al entrenamiento.

    Rechaza lotes que no traigan etiquetas reales con un mensaje claro.
    """
    if objetivo not in lote.columns:
        raise ValueError(
            f"Lote rechazado: falta la columna objetivo '{objetivo}' con las etiquetas reales. "
            "En producción las etiquetas pueden llegar con retraso; verifique la llegada de etiquetas antes de evaluar event drift."
        )

    valores_lote = lote[objetivo].dropna()
    if valores_lote.empty:
        raise ValueError(
            f"Lote rechazado: la columna objetivo '{objetivo}' no contiene valores válidos (todos son nulos). "
            "En producción las etiquetas pueden llegar con retraso."
        )

    if objetivo not in datos_referencia.columns or datos_referencia[objetivo].dropna().empty:
        raise ValueError(f"Los datos de referencia no contienen la variable objetivo '{objetivo}'")

    valores_ref = datos_referencia[objetivo].dropna()

    # Prueba estadística Kolmogorov-Smirnov de 2 muestras
    ks_res = ks_2samp(valores_ref, valores_lote)
    estadistico = float(ks_res.statistic)
    p_valor = float(ks_res.pvalue)

    media_ref = float(valores_ref.mean())
    media_lote = float(valores_lote.mean())
    dif_pct = float(((media_lote - media_ref) / media_ref) * 100.0) if media_ref != 0 else 0.0

    hay_drift = bool(p_valor < umbral_p_valor)
    hay_alerta = hay_drift

    mensaje = (
        f"Drift de evento (target '{objetivo}'): "
        f"p-valor={p_valor:.6f} (umbral={umbral_p_valor}), estadístico KS={estadistico:.4f}. "
        f"Media referencia: {media_ref:,.2f} | Media lote: {media_lote:,.2f} ({dif_pct:+.2f}%). "
        f"Estado: {'ALERTA DE DRIFT DETECTADA' if hay_drift else 'DISTRIBUCIÓN ESTABLE'}."
    )

    return ResultadoDriftTarget(
        variable_objetivo=objetivo,
        media_referencia=round(media_ref, 2),
        media_lote=round(media_lote, 2),
        diferencia_media_pct=round(dif_pct, 2),
        estadistico_ks=round(estadistico, 4),
        p_valor=float(p_valor),
        umbral_p_valor=umbral_p_valor,
        hay_drift=hay_drift,
        hay_alerta=hay_alerta,
        mensaje=mensaje,
    )


def guardar_reporte_features(
    resultado: ResultadoDriftFeatures,
    reporte_run: Any,
    carpeta_salida: str | Path,
    identificador_lote: str,
) -> tuple[Path, Path]:
    """Guarda reporte HTML interactivo y resumen JSON del drift de features."""
    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta_html = carpeta / f"reporte_features_{identificador_lote}.html"
    ruta_json = carpeta / f"reporte_features_{identificador_lote}.json"

    # Guardar reporte HTML de Evidently
    if hasattr(reporte_run, "save_html"):
        reporte_run.save_html(str(ruta_html))

    # Guardar resumen en formato JSON
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(asdict(resultado), f, indent=2, ensure_ascii=False)

    return ruta_html, ruta_json


def guardar_reporte_target(
    resultado: ResultadoDriftTarget,
    carpeta_salida: str | Path,
    identificador_lote: str,
) -> Path:
    """Guarda resumen JSON del drift de la variable objetivo."""
    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta_json = carpeta / f"reporte_target_{identificador_lote}.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(asdict(resultado), f, indent=2, ensure_ascii=False)

    return ruta_json
