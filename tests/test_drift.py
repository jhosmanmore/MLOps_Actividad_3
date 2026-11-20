"""Pruebas unitarias de la lógica de detección y emulación de drift (features y target).

Se ejecutan de manera aislada sin requerir servicios de MLflow ni Prefect.
"""
import pandas as pd
import pytest

from src.config import cargar_config
from src.drift import (
    aplicar_drift_emulado,
    detectar_drift_features,
    detectar_drift_target,
    dividir_referencia_y_prueba,
    guardar_reporte_features,
    guardar_reporte_target,
    tomar_lote_prueba,
)
from src.generate_data import generate_car_sales_data


@pytest.fixture(scope="module")
def datos_sinteticos():
    """Dataset base sintético pequeño para pruebas rápidas de drift."""
    return generate_car_sales_data(num_records=15000, random_state=42, output_path=None)


@pytest.fixture(scope="module")
def particiones(datos_sinteticos):
    df_train, df_test = dividir_referencia_y_prueba(datos_sinteticos, test_size=0.2, semilla=42)
    return df_train, df_test


# Pruebas de emulación de drift 

def test_nivel_drift_cero_no_modifica_datos(particiones):
    _, df_test = particiones
    lote = tomar_lote_prueba(df_test, tamano_lote=100, semilla=10)
    lote_drift = aplicar_drift_emulado(lote, nivel=0.0)
    pd.testing.assert_frame_equal(lote, lote_drift)


def test_nivel_drift_modifica_features_y_target(particiones):
    _, df_test = particiones
    lote = tomar_lote_prueba(df_test, tamano_lote=100, semilla=10)
    lote_drift = aplicar_drift_emulado(lote, nivel=0.6, semilla=42)

    # 1. El kilometraje medio debe aumentar significativamente
    assert lote_drift["kilometraje"].mean() > lote["kilometraje"].mean() + 40000

    # 2. Los años de modelo deben ser menores (vehículos más viejos)
    assert lote_drift["año_modelo"].mean() < lote["año_modelo"].mean()

    # 3. El precio de venta (target) debe cambiar su distribución (caída en media)
    assert lote_drift["precio_venta"].mean() < lote["precio_venta"].mean() * 0.85


def test_nivel_drift_invalido_lanza_error(particiones):
    _, df_test = particiones
    lote = tomar_lote_prueba(df_test, tamano_lote=50, semilla=1)
    with pytest.raises(ValueError, match="El nivel de drift debe estar entre 0.0 y 1.0"):
        aplicar_drift_emulado(lote, nivel=-0.1)

    with pytest.raises(ValueError, match="El nivel de drift debe estar entre 0.0 y 1.0"):
        aplicar_drift_emulado(lote, nivel=1.5)


# Pruebas de drift de features con Evidently 

def test_drift_features_compara_lote_0_y_0_6(particiones):
    df_train, df_test = particiones
    features = ["año_modelo", "kilometraje", "precio_lista", "marca", "tipo_vehiculo"]

    lote_base = tomar_lote_prueba(df_test, tamano_lote=200, semilla=42)
    lote_0_0 = aplicar_drift_emulado(lote_base, nivel=0.0, semilla=42)
    lote_0_6 = aplicar_drift_emulado(lote_base, nivel=0.6, semilla=42)

    umbral_alerta = 0.3

    res_0_0, _ = detectar_drift_features(df_train, lote_0_0, features, umbral_fraccion=umbral_alerta)
    res_0_6, _ = detectar_drift_features(df_train, lote_0_6, features, umbral_fraccion=umbral_alerta)

    # Comprobación requerida: lotes 0.0 y 0.6 producen reportes distintos
    assert res_0_0.fraccion_drift != res_0_6.fraccion_drift
    assert res_0_6.features_con_drift > res_0_0.features_con_drift

    # Solo el lote 0.6 supera el umbral de alerta (30%)
    assert res_0_0.hay_alerta is False
    assert res_0_6.hay_alerta is True


def test_guardar_reportes_features(particiones, tmp_path):
    df_train, df_test = particiones
    features = ["año_modelo", "kilometraje", "marca"]
    lote = tomar_lote_prueba(df_test, tamano_lote=50, semilla=1)

    res, reporte_run = detectar_drift_features(df_train, lote, features)
    ruta_html, ruta_json = guardar_reporte_features(res, reporte_run, tmp_path, "prueba_123")

    assert ruta_html.exists()
    assert ruta_json.exists()
    assert ruta_html.stat().st_size > 0
    assert ruta_json.stat().st_size > 0


# Pruebas de drift de variable objetivo/evento 

def test_rechaza_lote_sin_columna_objetivo(particiones):
    df_train, df_test = particiones
    lote_sin_etiquetas = tomar_lote_prueba(df_test, tamano_lote=50).drop(columns=["precio_venta"])

    with pytest.raises(ValueError, match="Lote rechazado: falta la columna objetivo 'precio_venta'"):
        detectar_drift_target(df_train, lote_sin_etiquetas, objetivo="precio_venta")


def test_rechaza_lote_con_etiquetas_nulas(particiones):
    df_train, df_test = particiones
    lote_nulo = tomar_lote_prueba(df_test, tamano_lote=50)
    lote_nulo["precio_venta"] = None

    with pytest.raises(ValueError, match="Lote rechazado: la columna objetivo 'precio_venta' no contiene valores válidos"):
        detectar_drift_target(df_train, lote_nulo, objetivo="precio_venta")


def test_drift_target_detecta_cambio_en_0_6_pero_no_en_0_0(particiones):
    df_train, df_test = particiones
    lote_base = tomar_lote_prueba(df_test, tamano_lote=150, semilla=42)

    lote_0_0 = aplicar_drift_emulado(lote_base, nivel=0.0)
    lote_0_6 = aplicar_drift_emulado(lote_base, nivel=0.6, objetivo="precio_venta", semilla=42)

    res_0_0 = detectar_drift_target(df_train, lote_0_0, objetivo="precio_venta", umbral_p_valor=0.05)
    res_0_6 = detectar_drift_target(df_train, lote_0_6, objetivo="precio_venta", umbral_p_valor=0.05)

    # 0.0 no debe tener drift significativo
    assert res_0_0.p_valor > 0.05
    assert res_0_0.hay_drift is False
    assert res_0_0.hay_alerta is False

    # 0.6 tiene drift evidente (p-valor muy bajo, alerta activada)
    assert res_0_6.p_valor < 0.05
    assert res_0_6.hay_drift is True
    assert res_0_6.hay_alerta is True
    assert abs(res_0_6.diferencia_media_pct) > 10.0


def test_guardar_reporte_target(particiones, tmp_path):
    df_train, df_test = particiones
    lote = tomar_lote_prueba(df_test, tamano_lote=50, semilla=1)
    res = detectar_drift_target(df_train, lote, objetivo="precio_venta")
    ruta_json = guardar_reporte_target(res, tmp_path, "prueba_target_123")

    assert ruta_json.exists()
    assert ruta_json.stat().st_size > 0
