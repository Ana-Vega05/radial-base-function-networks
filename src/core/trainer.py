import numpy as np
from typing import Callable, Optional
from core.rbf_model import RBFNetwork

def entrenar_rbf(X_train: np.ndarray, Yd_train: np.ndarray, config, min_X: float, max_X: float, verbose: bool = False, on_iter_progress: Optional[Callable[[int, int, float], None]] = None,):
    """
    Entrena la red RBF con aumento iterativo de centros.

    Algoritmo:
    - Inicializar n_centros centros aleatorios en [min_X, max_X]
    - Calcular D, FA, A, W por pseudoinversa
    - Calcular EG
    - Si EG <= error_optimo → converge y termina
    - Si no → incrementar n_centros y repetir (hasta max_iteraciones)"""
    n_centros_actual = config.n_centros
    historial_EG = []
    historial_n_centros = []
    convergio = False
    modelo = None
    EG = float("inf")

    sep = "═" * 60
    rng = np.random.default_rng(config.random_state)

    for iteracion in range(config.max_iteraciones):
        print(f"\n{sep}")
        print(f"\nIteración {iteracion + 1}/{config.max_iteraciones} con n_centros = {n_centros_actual}")
        print(sep)

        centros = rng.uniform(min_X, max_X, size=(n_centros_actual, config.n_entradas),)
        modelo = RBFNetwork(n_entradas=config.n_entradas, n_salidas=config.n_salidas, centros=centros)
        modelo.fit(X_train, Yd_train, verbose=verbose)

        EG = modelo.error_general(X_train, Yd_train)
        historial_EG.append(EG)
        historial_n_centros.append(n_centros_actual)

        print(f"  Error general (EG): {EG:.6f} (Error óptimo o umbral: {config.error_optimo:.6f})")

        if on_iter_progress is not None:
            on_iter_progress(iteracion + 1, config.max_iteraciones, EG)

        if EG <= config.error_optimo:
            print(f"  Convergencia alcanzada en iteración {iteracion + 1}.")
            convergio = True
            break
        else:
            print(f"  No se alcanzó el error óptimo. Incrementando n_centros a {n_centros_actual + 1}.")
            n_centros_actual += 1

    if not convergio:
        print(f"\n{sep}")
        print(f"\n No se alcanzó el error óptimo en {config.max_iteraciones} iteraciones.")
        print(f" Último EG: {EG:.6f} (umbral: {config.error_optimo:.6f})")

    print(sep)
    return modelo, historial_EG, historial_n_centros, convergio
