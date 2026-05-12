# -*- coding: utf-8 -*-
"""Persistencia de modelos RBF."""

import json
import numpy as np
from core.rbf_model import RBFNetwork

def save_model(model: RBFNetwork, config_dict: dict, info: dict, filepath: str):
    """
    Guarda el modelo y la configuración en un archivo JSON.
    Los arrays se convierten a listas para serialización.
    """
    data = {
        "centros": model.centros.tolist(),
        "pesos": model.pesos.tolist(),
        "n_entradas": model.n_entradas,
        "n_salidas": model.n_salidas,
        "config": config_dict,
        "min_X": info.get("min_X", 0.0),
        "max_X": info.get("max_X", 1.0),
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_model(filepath: str) -> tuple:
    """
    Carga un modelo guardado y retorna (modelo, config_dict, info).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    centros = np.array(data["centros"])
    pesos = np.array(data["pesos"])
    n_entradas = data["n_entradas"]
    n_salidas = data["n_salidas"]
    config_dict = data["config"]
    info = {
        "min_X": data.get("min_X", 0.0),
        "max_X": data.get("max_X", 1.0),
    }

    modelo = RBFNetwork(n_entradas=n_entradas, n_salidas=n_salidas,
                        centros=centros, pesos=pesos)
    return modelo, config_dict, info