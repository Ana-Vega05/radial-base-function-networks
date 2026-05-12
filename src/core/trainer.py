import numpy as np
from typing import Callable, Optional
from src.core.rbf_model import RBFNetwork

def entrenar_rbf(X_train: np.ndarray, Yd_train: np.ndarray, config, min_X: float, max_X: float, verbose: bool = False, on_iter_progress: Optional[Callable[[int, int, float], None]] = None,):
    """
    Entrena la red RBF con aumento iterativo de centros.

    Algoritmo:
    - Inicializar n_centros centros aleatorios en [min_X, max_X]
    - Calcular D, FA, A, W por pseudoinversa
    - Calcular EG
    - Si EG <= error_optimo → converge y termina
    """
    n_centros_actual = config.n_centros
    historial_EG = []
    historial_n_centros = []
    convergio = False
    mejor_modelo        = None
    mejor_EG            = float("inf")
    mejor_iter          = -1
    EG = float("inf")
    

    sep = "═" * 60
    rng = np.random.default_rng(config.random_state)

    for iteracion in range(config.max_iteraciones):
        print(f"\n{sep}")
        print(f"\nIteración {iteracion + 1}/{config.max_iteraciones} con n_centros = {config.n_centros}")
        print(sep)

        # Inicializar modelo con centros aleatorios dentro del rango  [min_X, max_X]
        centros = rng.uniform( min_X, max_X, size=(config.n_centros, config.n_entradas),)
        # Crear y entrenar modelo
        modelo = RBFNetwork(n_entradas=config.n_entradas, n_salidas=config.n_salidas, centros=centros)
        modelo.fit(X_train, Yd_train, verbose=verbose)

        
        # Calcular error general en entrenamiento
        EG = modelo.error_general(X_train, Yd_train)
        historial_EG.append(EG)
        historial_n_centros.append(config.n_centros)
        
        print(f"  Error general (EG): {EG:.6f} (Error óptimo o umbral: {config.error_optimo:.6f})")
        if EG < mejor_EG:
            mejor_EG     = EG
            mejor_modelo = modelo
            mejor_iter   = iteracion + 1
            print(f" Mejor hasta ahora — EG = {EG:.6f}")
        else:
            print(f"    EG = {EG:.6f}  (mejor: {mejor_EG:.6f} en intento {mejor_iter})")

        print(f"    Error óptimo        : {config.error_optimo:.6f}")
        # Notificar progreso a la GUI (si hay callback)
        if on_iter_progress is not None:
            on_iter_progress(iteracion + 1, config.max_iteraciones, EG)

        # Verificar convergencia
        if EG <= config.error_optimo:
            print(f"  Convergencia alcanzadaen iteración {iteracion + 1}.")
            convergio = True
            break
        else: 
            print(f"  No se alcanzó el error óptimo. Incrementando n_centros para la siguiente iteración.")

    if not convergio:
        print(f"\n{sep}")
        print(f"\n No se alcanzó el error óptimo en {config.max_iteraciones} iteraciones.")
        print(f" Último EG: {EG:.6f} (umbral: {config.error_optimo:.6f})")

    
    print(sep)
    return mejor_modelo, historial_EG, historial_n_centros, convergio