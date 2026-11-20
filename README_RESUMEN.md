# Resumen de cambios del repositorio

Actividad 3 MLOps

## Archivos modificados

| Archivo | Tipo | Resumen del cambio |
|---|---|---|
| `configs/config.yaml` | Configuración | Se incorporó la configuración de inferencia por lotes y de monitoreo de drift: carpeta de lotes y predicciones, nivel de drift emulado, umbrales de alertas y rutas de reportes. |
| `flows/inferencia_flow.py` | Pipeline | Se extendió el flujo de inferencia para tomar lotes de prueba, aplicar drift emulado, guardar las predicciones y conservar una copia del lote utilizado para su posterior análisis. |
| `pyproject.toml` | Dependencias | Se actualizaron las dependencias del proyecto y se agregó Evidently y SciPy para el monitoreo de drift, además de mantener la configuración de pytest. |
| `tests/test_datos.py` | Pruebas | Se ampliaron las pruebas de generación y validación de datos: reproducibilidad, estructura, rangos, valores nulos, tipos y regeneración del archivo CSV. |

## Archivos nuevos

| Archivo | Tipo | Resumen del contenido |
|---|---|---|
| `flows/drift_features_flow.py` | Pipeline Prefect | Compara la distribución de las variables de entrada frente a la referencia de entrenamiento mediante Evidently y genera reportes HTML y JSON. |
| `flows/drift_target_flow.py` | Pipeline Prefect | Evalúa el drift de la variable objetivo `precio_venta`, valida la presencia de etiquetas reales y genera un resumen estadístico en JSON. |
| `poetry.toml` | Configuración | Configura Poetry para usar el entorno virtual dentro del proyecto y define el directorio de caché. |
| `src/drift.py` | Lógica de negocio | Implementa la partición de datos, toma de lotes, emulación de drift, detección de drift de features y target, y guardado de reportes. |
| `tests/test_drift.py` | Pruebas | Cubre la emulación de drift, la detección de drift con Evidently, el drift estadístico del objetivo y la generación de reportes. |
| `tests/test_inferencia.py` | Pruebas | Valida el tamaño y las columnas de los lotes, el cálculo de predicciones, los errores absolutos y la conservación de la estructura al aplicar drift. |

## Funcionalidades agregadas

- Emulación configurable de drift en `kilometraje`, `año_modelo`, `precio_lista`, `tipo_vehiculo` y `precio_venta`.
- Conservación de los lotes puntuados para ser evaluados posteriormente.
- Detección de data drift con Evidently y umbral configurable del 30 % de features por defecto.
- Detección de event drift en el objetivo mediante la prueba estadística de Kolmogorov-Smirnov.
- Generación de reportes HTML y JSON organizados por tipo de drift.
- Pruebas unitarias independientes de MLflow y Prefect para la lógica de datos y drift.

## Ejecución

Desde la raíz del repositorio:

```powershell
poetry install
poetry run python flows/inferencia_flow.py
poetry run python flows/drift_features_flow.py
poetry run python flows/drift_target_flow.py
poetry run pytest
```
