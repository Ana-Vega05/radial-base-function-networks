import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

def load_json_dataset(config) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Carga un dataset desde un archivo JSON.
    Soporta dos formatos:
    1. Raíz es una lista de registros.
    2. Objeto con clave "data" que contiene la lista.
    """
    json_path = os.path.join(config.raw_data_path, f"{config.dataset_name}.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"No se encontró el archivo JSON: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    # Detectar formato
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict) and "data" in raw:
        records = raw["data"]
        # Opcional: mostrar features
        if "features" in raw:
            print(f"Features en el JSON: {raw['features']}")
            # Validar que coincidan con las columnas de entrada configuradas
            if set(raw["features"]) != set(config.input_columns):
                print("Las columnas de entrada configuradas no coinciden exactamente con las del JSON.")
    else:
        raise ValueError("Formato de JSON no reconocido. Se esperaba una lista o un objeto con clave 'data'.")
    
    if len(records) == 0:
        raise ValueError("El conjunto de datos está vacío.")
    
    df = pd.DataFrame(records)
    
    # Si el registro tiene la estructura {"input": [...], "output": valor}
    # entonces necesitamos aplanar "input" en columnas separadas
    if "input" in df.columns and "output" in df.columns:
        # Expandir la lista de inputs en columnas x1, x2, ...
        input_df = pd.DataFrame(df["input"].tolist(), columns=config.input_columns)
        output_df = df[["output"]].rename(columns={"output": config.target_column})
        df = pd.concat([input_df, output_df], axis=1)
    
    # Validar columnas requeridas
    missing_inputs = set(config.input_columns) - set(df.columns)
    if missing_inputs:
        raise ValueError(f"Faltan columnas de entrada en el dataset: {missing_inputs}")
    if config.target_column not in df.columns:
        raise ValueError(f"Falta la columna objetivo '{config.target_column}' en el dataset.")
    
    X = df[config.input_columns].values.astype(np.float64)
    Yd = df[[config.target_column]].values.astype(np.float64)
    
    # Guardar CSV
    os.makedirs(config.processed_data_path, exist_ok=True)
    csv_path = os.path.join(config.processed_data_path, f"{config.dataset_name}.csv")
    df.to_csv(csv_path, index=False, sep=';', decimal=',')
    
    # Actualizar configuración
    config.n_entradas = X.shape[1]
    config.n_salidas = Yd.shape[1]
    config.n_patrones = X.shape[0]
    
    info = {
        "n_entradas": config.n_entradas,
        "n_salidas": config.n_salidas,
        "n_patrones": config.n_patrones,
        "min_X": float(np.min(X)),
        "max_X": float(np.max(X)),
        "estadisticas": df.describe(include='all').to_dict(),
        "csv_guardado": csv_path
    }
    
    return X, Yd, info