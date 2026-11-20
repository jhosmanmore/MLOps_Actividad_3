"""Pruebas de la lógica de inferencia (cálculo de predicciones, toma de lotes y copias)."""
import numpy as np
import pandas as pd
import pytest

from src.drift import aplicar_drift_emulado, dividir_referencia_y_prueba, tomar_lote_prueba
from src.generate_data import generate_car_sales_data


class ModeloFalso:
    """Modelo simple para probar la lógica de inferencia sin requerir MLflow."""
    def predict(self, X):
        return np.array([20000.0] * len(X))


@pytest.fixture(scope="module")
def datos_prueba():
    df = generate_car_sales_data(num_records=15000, random_state=42, output_path=None)
    _, df_test = dividir_referencia_y_prueba(df, test_size=0.2, semilla=42)
    return df_test


def test_tomar_lote_prueba_tamano_y_columnas(datos_prueba):
    lote = tomar_lote_prueba(datos_prueba, tamano_lote=50, semilla=123)
    assert len(lote) == 50
    assert "precio_venta" in lote.columns
    assert "kilometraje" in lote.columns


def test_predecir_agrega_columnas_esperadas(datos_prueba):
    lote = tomar_lote_prueba(datos_prueba, tamano_lote=20, semilla=1)
    champion = {
        "modelo": ModeloFalso(),
        "columnas": ["marca", "tipo_vehiculo", "año_modelo", "kilometraje"],
    }
    
    # Simular la lógica de inferencia
    resultado = lote.copy()
    resultado["precio_predicho"] = champion["modelo"].predict(lote[champion["columnas"]]).round(2)
    resultado["error_absoluto"] = (resultado["precio_predicho"] - resultado["precio_venta"]).abs().round(2)

    assert "precio_predicho" in resultado.columns
    assert "error_absoluto" in resultado.columns
    assert (resultado["error_absoluto"] >= 0).all()
    assert (resultado["precio_predicho"] == 20000.0).all()


def test_lote_con_drift_conserva_columnas(datos_prueba):
    lote = tomar_lote_prueba(datos_prueba, tamano_lote=30, semilla=5)
    columnas_originales = list(lote.columns)

    for nivel in [0.0, 0.3, 0.6]:
        lote_drift = aplicar_drift_emulado(lote, nivel=nivel, semilla=42)
        assert list(lote_drift.columns) == columnas_originales
        assert len(lote_drift) == 30
        assert not lote_drift.isna().any().any()
