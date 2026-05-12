from typing import Optional

from sklearn.model_selection import train_test_split
import numpy as np

def split_dataset(X: np.ndarray, Yd: np.ndarray, tipo_particion: str, random_state: Optional[int]) -> dict:
    """
    Divide el dataset en entrenamiento, validación y prueba.
    Soporta Yd binario (n, 1) y Yd one-hot multiclase (n, n_clases).
    En ambos casos aplica stratify para mantener distribución de clases.
    
    Args:
        X: matriz de entradas (n_patrones, n_entradas)
        Yd: matriz de salidas (n_patrones, n_salidas)
        tipo_particion: "80-10-10" o "70-15-15"
        random_state: semilla para reproducibilidad
    
    Returns:
        Diccionario con llaves:
            X_train, Yd_train, X_val, Yd_val, X_test, Yd_test
    """
    if tipo_particion == "80-10-10":
        train_size = 0.8
        val_size = 0.1
        test_size = 0.1
    elif tipo_particion == "70-15-15":
        train_size = 0.7
        val_size = 0.15
        test_size = 0.15
    else:
        raise ValueError(f"Tipo de partición no soportado: {tipo_particion}")
    
    # Etiquetas de clase para stratify
    # Binario  → ravel directo
    # One-hot  → argmax para recuperar la clase entera
    if Yd.shape[1] == 1:
        stratify_labels = Yd.ravel().astype(int)
    else:
        stratify_labels = np.argmax(Yd, axis=1)
    
    # Primera división: train+val vs test
    X_temp, X_test, Yd_temp, Yd_test, strat_temp, _ = train_test_split(
        X, Yd, stratify_labels, test_size=test_size, random_state=random_state, stratify=stratify_labels
)
    
    # Segunda división: train vs val
    val_relative_size = val_size / (train_size + val_size)  # Proporción de val dentro del bloque temp
    X_train, X_val, Yd_train, Yd_val = train_test_split(
        X_temp, Yd_temp, test_size=val_relative_size, random_state=random_state, stratify=strat_temp
    )
    
    return {
        "X_train": X_train,
        "Yd_train": Yd_train,
        "X_val": X_val,
        "Yd_val": Yd_val,
        "X_test": X_test,
        "Yd_test": Yd_test
    }