"""Pipeline de inferencia: valora un lote de vehículos de prueba con drift emulado usando el modelo champion.

Pasos: cargar champion -> tomar lote de prueba con drift -> predecir -> guardar lote, predicciones y registrar en MLflow.

El nivel de drift está controlado por el parámetro de configuración `inferencia.nivel_drift`
(de 0.0 a 1.0) en configs/config.yaml. El usuario no lo introduce por línea de comandos.

Ejecución (desde la raíz del repositorio, con los servicios levantados):
    poetry run python flows/inferencia_flow.py
"""
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # permite importar src/

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402

from src.config import cargar_config, ruta_absoluta  # noqa: E402
from src.drift import aplicar_drift_emulado, dividir_referencia_y_prueba, tomar_lote_prueba  # noqa: E402
from src.load_data import cargar_datos  # noqa: E402


@task(name="cargar-champion", cache_policy=NO_CACHE)
def cargar_champion(config: dict) -> dict:
    """Carga el modelo con alias champion y las columnas con las que se entrenó."""
    logger = get_run_logger()
    nombre = config["registro"]["nombre_modelo"]
    uri = f"models:/{nombre}@champion"
    version = MlflowClient().get_model_version_by_alias(nombre, "champion")
    modelo = mlflow.sklearn.load_model(uri)
    columnas = mlflow.models.get_model_info(uri).signature.inputs.input_names()
    logger.info(f"Champion: {nombre} versión {version.version} | variables: {columnas}")
    return {"modelo": modelo, "version": version.version, "run_id": version.run_id, "columnas": columnas}


@task(name="tomar-lote-con-drift", cache_policy=NO_CACHE)
def tomar_lote(config: dict, nivel_drift_override: float | None = None) -> tuple[pd.DataFrame, int, float]:
    """Toma un lote de la partición de prueba del dataset y le aplica drift emulado según la configuración."""
    logger = get_run_logger()
    semilla = random.randint(0, 1_000_000)

    # 1. Cargar el dataset completo y separar la partición de prueba (idéntica al entrenamiento)
    df = cargar_datos(config["datos"])
    _, df_test = dividir_referencia_y_prueba(
        df,
        test_size=config["division"]["test_size"],
        semilla=config["division"]["semilla"],
    )

    # 2. Tomar el lote de la partición de prueba
    tamano_lote = config["inferencia"]["tamano_lote"]
    lote_base = tomar_lote_prueba(df_test, tamano_lote=tamano_lote, semilla=semilla)

    # 3. Nivel de drift: tomado de la configuración central
    nivel_drift = float(
        nivel_drift_override if nivel_drift_override is not None else config["inferencia"].get("nivel_drift", 0.0)
    )

    # 4. Aplicar emulación de drift antes de la predicción
    lote_con_drift = aplicar_drift_emulado(
        lote_base,
        nivel=nivel_drift,
        objetivo=config["datos"]["objetivo"],
        semilla=semilla,
    )

    logger.info(
        f"Lote de {len(lote_con_drift)} vehículos extraído de la partición de prueba (semilla {semilla}) | "
        f"Nivel de drift emulado aplicado: {nivel_drift:.2f}"
    )
    return lote_con_drift, semilla, nivel_drift


@task(name="predecir", cache_policy=NO_CACHE)
def predecir(champion: dict, lote: pd.DataFrame, objetivo: str) -> pd.DataFrame:
    """Añade al lote el precio predicho y el error frente al precio real."""
    logger = get_run_logger()
    resultado = lote.copy()
    resultado["precio_predicho"] = champion["modelo"].predict(lote[champion["columnas"]]).round(2)
    resultado["error_absoluto"] = (resultado["precio_predicho"] - resultado[objetivo]).abs().round(2)
    logger.info(
        f"Precio medio predicho: {resultado['precio_predicho'].mean():,.2f} USD | "
        f"Error medio del lote (MAE): {resultado['error_absoluto'].mean():,.2f} USD"
    )
    return resultado


@task(name="guardar-y-registrar", cache_policy=NO_CACHE)
def guardar_y_registrar(
    predicciones: pd.DataFrame,
    champion: dict,
    semilla: int,
    nivel_drift: float,
    config: dict,
) -> tuple[Path, Path]:
    """Guarda predicciones y una copia del lote puntuado para que los pipelines de drift la lean.

    Registra la ejecución y artefactos en MLflow.
    """
    logger = get_run_logger()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nivel_str = f"{nivel_drift:.1f}".replace(".", "_")

    # Carpetas de salida
    carpeta_pred = ruta_absoluta(config["inferencia"]["carpeta_salida"])
    carpeta_lotes = ruta_absoluta(config["inferencia"].get("carpeta_lotes", "data/lotes"))
    carpeta_pred.mkdir(parents=True, exist_ok=True)
    carpeta_lotes.mkdir(parents=True, exist_ok=True)

    # 1. Guardar predicciones
    ruta_prediccion = carpeta_pred / f"prediccion_{timestamp}_drift_{nivel_str}.csv"
    predicciones.to_csv(ruta_prediccion, index=False, lineterminator="\n")

    # 2. Guardar copia del lote puntuado (para lectura de pipelines de drift)
    ruta_lote = carpeta_lotes / f"lote_{timestamp}_drift_{nivel_str}.csv"
    ruta_lote_reciente = carpeta_lotes / "lote_mas_reciente.csv"
    predicciones.to_csv(ruta_lote, index=False, lineterminator="\n")
    predicciones.to_csv(ruta_lote_reciente, index=False, lineterminator="\n")

    logger.info(f"Predicciones guardadas en: {ruta_prediccion}")
    logger.info(f"Copia del lote con drift guardada en: {ruta_lote} y {ruta_lote_reciente}")

    # 3. Registrar en MLflow
    mlflow.set_experiment(config["mlflow"]["experimento"])
    run_name = f"{config['mlflow']['nombre_run_inferencia']}-{timestamp}-drift{nivel_str}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "fase": "inferencia",
            "modelo_version": champion["version"],
            "modelo_run_id": champion["run_id"],
            "nivel_drift": str(nivel_drift),
        })
        mlflow.log_params({
            "tamano_lote": len(predicciones),
            "semilla_lote": semilla,
            "nivel_drift": nivel_drift,
        })
        mlflow.log_metrics({
            "filas_predichas": len(predicciones),
            "precio_medio_predicho": float(predicciones["precio_predicho"].mean()),
            "mae_lote": float(predicciones["error_absoluto"].mean()),
        })
        mlflow.log_artifact(str(ruta_prediccion), artifact_path="predicciones")
        mlflow.log_artifact(str(ruta_lote), artifact_path="lotes_evaluados")

    return ruta_prediccion, ruta_lote


@flow(name="inferencia-precio-vehiculos", log_prints=True)
def inferencia_flow(nivel_drift: float | None = None) -> tuple[str, str]:
    config = cargar_config()
    champion = cargar_champion(config)
    lote, semilla, nivel_aplicado = tomar_lote(config, nivel_drift_override=nivel_drift)
    predicciones = predecir(champion, lote, config["datos"]["objetivo"])
    ruta_pred, ruta_lote = guardar_y_registrar(predicciones, champion, semilla, nivel_aplicado, config)
    print(f"Inferencia completada (versión {champion['version']}, drift={nivel_aplicado:.1f}): {ruta_pred.name}")
    return str(ruta_pred), str(ruta_lote)


if __name__ == "__main__":
    inferencia_flow()
