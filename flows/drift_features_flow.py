"""Pipeline de detección de drift en variables de entrada (features) con Evidently.

Compara la distribución de cada variable de entrada de un lote puntuado frente a los
datos de entrenamiento (referencia).
Si la fracción de variables con drift supera el umbral configurado (ej. 30 %), emite una alerta.
Guarda un reporte interactivo en HTML y un resumen en JSON por cada lote evaluado.

Ejecución (con servicios levantados):
    poetry run python flows/drift_features_flow.py
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402

from src.config import cargar_config, cargar_features, ruta_absoluta  # noqa: E402
from src.drift import detectar_drift_features, dividir_referencia_y_prueba, guardar_reporte_features  # noqa: E402
from src.features import resolver_variables  # noqa: E402
from src.load_data import cargar_datos  # noqa: E402


@task(name="cargar-referencia-entrenamiento", cache_policy=NO_CACHE)
def cargar_referencia(config: dict) -> pd.DataFrame:
    """Obtiene el conjunto de datos de entrenamiento para usar como referencia de distribución."""
    logger = get_run_logger()
    df_completo = cargar_datos(config["datos"])
    df_train, _ = dividir_referencia_y_prueba(
        df_completo,
        test_size=config["division"]["test_size"],
        semilla=config["division"]["semilla"],
    )
    logger.info(f"Datos de referencia (entrenamiento) cargados: {len(df_train)} filas")
    return df_train


@task(name="cargar-lote-a-evaluar", cache_policy=NO_CACHE)
def cargar_lote(ruta_lote: str | Path | None, config: dict) -> tuple[pd.DataFrame, str]:
    """Carga el lote a evaluar (por defecto el más reciente en la carpeta de lotes)."""
    logger = get_run_logger()
    if ruta_lote is None:
        carpeta_lotes = ruta_absoluta(config["inferencia"].get("carpeta_lotes", "data/lotes"))
        ruta_archivo = carpeta_lotes / "lote_mas_reciente.csv"
        if not ruta_archivo.exists():
            # Buscar el archivo de lote más reciente en la carpeta
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
            raise FileNotFoundError(f"El archivo de lote especificado no existe: {ruta_archivo}")

    lote = pd.read_csv(ruta_archivo)
    identificador = ruta_archivo.stem.replace("lote_", "").replace("prediccion_", "")
    logger.info(f"Lote cargado: {len(lote)} registros desde {ruta_archivo.name}")
    return lote, identificador


@task(name="evaluar-drift-features", cache_policy=NO_CACHE)
def evaluar_drift(
    df_referencia: pd.DataFrame,
    lote: pd.DataFrame,
    config: dict,
    config_features: dict,
) -> tuple[Any, Any]:
    """Ejecuta Evidently para comparar la distribución de features entre entrenamiento y el lote."""
    logger = get_run_logger()
    variables = resolver_variables(config_features, list(df_referencia.columns))
    columnas_a_evaluar = variables.todas

    umbral_fraccion = float(config.get("drift", {}).get("umbral_fraccion_drift", 0.3))
    umbral_p_valor = float(config.get("drift", {}).get("umbral_p_valor", 0.05))

    logger.info(f"Evaluando drift de features en {len(columnas_a_evaluar)} variables: {columnas_a_evaluar}")
    logger.info(f"Umbral de alerta: {umbral_fraccion:.1%} de variables con drift")

    resultado, reporte_run = detectar_drift_features(
        datos_referencia=df_referencia,
        lote=lote,
        columnas_features=columnas_a_evaluar,
        umbral_fraccion=umbral_fraccion,
        umbral_p_valor=umbral_p_valor,
    )

    if resultado.hay_alerta:
        logger.warning(f"⚠️ ALERTA DE DATA DRIFT: {resultado.mensaje}")
    else:
        logger.info(f"✅ Sin alerta de drift: {resultado.mensaje}")

    return resultado, reporte_run


@task(name="guardar-reporte-features", cache_policy=NO_CACHE)
def guardar_reporte(
    resultado: Any,
    reporte_run: Any,
    identificador: str,
    config: dict,
) -> tuple[Path, Path]:
    """Guarda reporte HTML de Evidently y resumen JSON en reports/drift_features/."""
    logger = get_run_logger()
    carpeta_reportes = ruta_absoluta(config.get("drift", {}).get("carpeta_reportes_features", "reports/drift_features"))
    ruta_html, ruta_json = guardar_reporte_features(
        resultado=resultado,
        reporte_run=reporte_run,
        carpeta_salida=carpeta_reportes,
        identificador_lote=identificador,
    )
    logger.info(f"Reporte HTML interactivo guardado en: {ruta_html}")
    logger.info(f"Resumen JSON guardado en: {ruta_json}")
    return ruta_html, ruta_json


@flow(name="deteccion-drift-features", log_prints=True)
def drift_features_flow(ruta_lote: str | None = None) -> dict:
    config = cargar_config()
    config_features = cargar_features()

    df_ref = cargar_referencia(config)
    lote, identificador = cargar_lote(ruta_lote, config)
    resultado, reporte_run = evaluar_drift(df_ref, lote, config, config_features)
    ruta_html, ruta_json = guardar_reporte(resultado, reporte_run, identificador, config)

    print("\n" + "=" * 60)
    print("REPORTE DE DRIFT DE FEATURES (EVIDENTLY)")
    print("=" * 60)
    print(resultado.mensaje)
    print(f"HTML: {ruta_html}")
    print(f"JSON: {ruta_json}")
    print("=" * 60 + "\n")

    return {
        "identificador": identificador,
        "fraccion_drift": resultado.fraccion_drift,
        "features_con_drift": resultado.features_con_drift,
        "hay_alerta": resultado.hay_alerta,
        "ruta_html": str(ruta_html),
        "ruta_json": str(ruta_json),
    }


if __name__ == "__main__":
    drift_features_flow()
