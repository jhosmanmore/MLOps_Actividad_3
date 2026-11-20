"""Pipeline de detección de drift en la variable objetivo (event drift).

Compara la distribución de la variable objetivo (`precio_venta`) del lote puntuado
frente a los datos de entrenamiento (referencia).
El lote debe traer las etiquetas reales (que en producción pueden llegar con retraso).
Rechaza los lotes sin etiquetas reales con un mensaje claro.
Si la prueba estadística detecta un cambio significativo en la distribución del objetivo, emite una alerta.

Ejecución (con servicios levantados):
    poetry run python flows/drift_target_flow.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402

from src.config import cargar_config, ruta_absoluta  # noqa: E402
from src.drift import (  # noqa: E402
    ResultadoDriftTarget,
    detectar_drift_target,
    dividir_referencia_y_prueba,
    guardar_reporte_target,
)
from src.load_data import cargar_datos  # noqa: E402


@task(name="cargar-referencia-target", cache_policy=NO_CACHE)
def cargar_referencia(config: dict) -> pd.DataFrame:
    """Obtiene los datos de entrenamiento para la referencia del objetivo."""
    logger = get_run_logger()
    df_completo = cargar_datos(config["datos"])
    df_train, _ = dividir_referencia_y_prueba(
        df_completo,
        test_size=config["division"]["test_size"],
        semilla=config["division"]["semilla"],
    )
    logger.info(f"Referencia de entrenamiento cargada: {len(df_train)} registros")
    return df_train


@task(name="cargar-lote-para-target", cache_policy=NO_CACHE)
def cargar_lote(ruta_lote: str | Path | None, config: dict) -> tuple[pd.DataFrame, str]:
    """Carga el lote a evaluar para análisis de target drift."""
    logger = get_run_logger()
    if ruta_lote is None:
        carpeta_lotes = ruta_absoluta(config["inferencia"].get("carpeta_lotes", "data/lotes"))
        ruta_archivo = carpeta_lotes / "lote_mas_reciente.csv"
        if not ruta_archivo.exists():
            lotes = sorted(carpeta_lotes.glob("lote_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not lotes:
                raise FileNotFoundError(
                    f"No se encontraron lotes para evaluar en {carpeta_lotes}. "
                    "Ejecute primero la pipeline de inferencia para generar un lote."
                )
            ruta_archivo = lotes[0]
    else:
        ruta_archivo = ruta_absoluta(ruta_lote)
        if not ruta_archivo.exists():
            raise FileNotFoundError(f"El archivo de lote no existe: {ruta_archivo}")

    lote = pd.read_csv(ruta_archivo)
    identificador = ruta_archivo.stem.replace("lote_", "").replace("prediccion_", "")
    logger.info(f"Lote cargado: {len(lote)} registros desde {ruta_archivo.name}")
    return lote, identificador


@task(name="evaluar-drift-target", cache_policy=NO_CACHE)
def evaluar_drift_objetivo(
    df_referencia: pd.DataFrame,
    lote: pd.DataFrame,
    config: dict,
) -> ResultadoDriftTarget:
    """Evalúa la hipótesis de cambio de distribución en el objetivo (KS-test).

    Rechaza el lote si no incluye las etiquetas reales de precio_venta.
    """
    logger = get_run_logger()
    objetivo = config["datos"]["objetivo"]
    umbral_p_valor = float(config.get("drift", {}).get("umbral_p_valor", 0.05))

    logger.info(f"Evaluando drift de variable objetivo ('{objetivo}') con umbral p-valor={umbral_p_valor}")

    try:
        resultado = detectar_drift_target(
            datos_referencia=df_referencia,
            lote=lote,
            objetivo=objetivo,
            umbral_p_valor=umbral_p_valor,
        )
    except ValueError as e:
        logger.error(f"❌ {e}")
        raise

    if resultado.hay_alerta:
        logger.warning(f"⚠️ ALERTA DE EVENT DRIFT: {resultado.mensaje}")
    else:
        logger.info(f"✅ Sin alerta de event drift: {resultado.mensaje}")

    return resultado


@task(name="guardar-reporte-target", cache_policy=NO_CACHE)
def guardar_reporte(
    resultado: ResultadoDriftTarget,
    identificador: str,
    config: dict,
) -> Path:
    """Guarda el reporte en JSON en reports/drift_target/."""
    logger = get_run_logger()
    carpeta = ruta_absoluta(config.get("drift", {}).get("carpeta_reportes_target", "reports/drift_target"))
    ruta_json = guardar_reporte_target(resultado, carpeta, identificador)
    logger.info(f"Reporte de drift de objetivo guardado en: {ruta_json}")
    return ruta_json


@flow(name="deteccion-drift-target", log_prints=True)
def drift_target_flow(ruta_lote: str | None = None) -> dict:
    config = cargar_config()
    df_ref = cargar_referencia(config)
    lote, identificador = cargar_lote(ruta_lote, config)
    resultado = evaluar_drift_objetivo(df_ref, lote, config)
    ruta_json = guardar_reporte(resultado, identificador, config)

    print("\n" + "=" * 60)
    print("REPORTE DE DRIFT DE VARIABLE OBJETIVO (EVENT DRIFT)")
    print("=" * 60)
    print(resultado.mensaje)
    print(f"Reporte guardado en: {ruta_json}")
    print("=" * 60 + "\n")

    return {
        "identificador": identificador,
        "objetivo": resultado.variable_objetivo,
        "p_valor": resultado.p_valor,
        "estadistico_ks": resultado.estadistico_ks,
        "hay_drift": resultado.hay_drift,
        "hay_alerta": resultado.hay_alerta,
        "ruta_json": str(ruta_json),
    }


if __name__ == "__main__":
    drift_target_flow()
