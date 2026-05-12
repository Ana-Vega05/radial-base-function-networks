import numpy as np

class RBFNetwork:
    """Red de Funciones de Base Radial (RBF) """
    def __init__(self, n_entradas: int, n_salidas: int, centros: np.ndarray = None, pesos: np.ndarray = None):
        self.n_entradas = n_entradas
        self.n_salidas = n_salidas # matriz (n_centros, n_entradas)
        self.centros = centros # (n_centros, n_entradas)
        self.pesos = pesos # (n_centros + 1, n_salidas)  incluye bias en fila 0

    @staticmethod
    def _activacion(omega: np.ndarray) -> np.ndarray:
        """Funcion de activacion (FA)"""
        resultado = np.zeros_like(omega, dtype=np.float64)
        mask = omega > 0
        resultado[mask] = omega[mask] ** 2* np.log(omega[mask])
        return resultado
    
    def _distancia_euclidiana(self, X: np.ndarray, centros: np.ndarray) -> np.ndarray:
        """
        Retorna matriz D de forma (n_patrones, n_centros). 
        D[i, j] = distancia euclidiana entre patrón i y centro j.
        """
        # diff: (n_patrones, n_centros, n_entradas)
        diff = X[:, np.newaxis, :] - centros[np.newaxis, :, :]
        # norm a lo largo del eje de entradas -> (n_patrones, n_centros)
        distancias = np.linalg.norm(diff, axis=2)
        return distancias
    
    def _construir_matriz_activacion(self, X, centros):
        """Construye la matriz de activación A para un conjunto de patrones X"""
        distancias = self._distancia_euclidiana(X, centros)  # (n_patrones, n_centros)
        FA = self._activacion(distancias)  # (n_patrones, n_centros)
        # Agregar columna de sesgo (bias)
        bias  = np.ones((FA.shape[0], 1))  # (n_patrones, 1)
        A = np.hstack([bias, FA])  # (n_patrones, n_centros + 1)
        return A, distancias, FA # devolvemos D y FA también para el modo verbose
    
    def fit(self, X_train: np.ndarray, Yd_train: np.ndarray, verbose: bool = False) -> 'RBFNetwork':
        """
        Calcula los pesos por pseudoinversa.
        verbose=True imprime:
        - Centros radiales R
        - Matriz de distancias D
        - Matriz de activaciones FA
        - Matriz de interpolación A
        - Sistema A·W = Yd
        - Pesos W  (Wo, W1…Wc)
        - Salidas calculadas Yr
        - Errores locales EL
        """
        if self.centros is None:
            raise ValueError("Los centros deben inicializarse antes de entrenar.")
        A, D, FA = self._construir_matriz_activacion(X_train, self.centros)
        # Pseudoinversa: W = pinv(A) * Yd
        self.pesos = np.linalg.pinv(A) @ Yd_train  # (n_centros + 1, n_salidas)
        if verbose:
            self._imprimir_detalle_entrenamiento(X_train, Yd_train, A, D, FA)
        return self
    
    def predict(self, X: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Realiza predicciones para un conjunto de patrones X"""
        if self.centros is None or self.pesos is None:
            raise ValueError("La red no ha sido entrenada.")
        A, D, FA= self._construir_matriz_activacion(X, self.centros)  # (n_patrones, n_centros + 1)
        Y_pred = A @ self.pesos  # (n_patrones, n_salidas)
        if verbose:
            self._imprimir_detalle_simulacion(X, A, D, FA, Y_pred)

        return Y_pred
    
    def error_general(self, X: np.ndarray, Yd: np.ndarray) -> float:
        """Calcula el error general E = (1/n) * sum((Yd - Y_pred)^2)"""
        Y_pred = self.predict(X)
        error = np.mean(np.abs(Yd - Y_pred))
        return error
    
    def _imprimir_detalle_entrenamiento(self, X_train, Yd_train, A, D, FA):
        n_centros = self.centros.shape[0]
        n_patrones = X_train.shape[0]

        sep = '-' * 60
        print(f"\n{sep}")
        print(f"DETALLE DEL ENTRENAMIENTO")
        print(sep)

        # Centros
        print(f"\nCentros radiales R [n_centros={n_centros}):\n{self.centros}\n x n_entradas={self.n_entradas}]:")
        for j, r in enumerate(self.centros):
            vals = " ".join(f"{v:.4f}" for v in r)
            print(f"R{j+1}: {vals}")
        
        #Distancias
        print(f"\nMatriz de distancias D [n_patrones={n_patrones} x n_centros={n_centros}]:")
        header = " Patron "+ " ".join(f"DP,R{j+1}" for j in range(n_centros))
        print(header)
        for i in range(n_patrones):
            fila = f"{i+1: >3}"
            for j in range(n_centros):
                fila += f" {D[i, j]:8.4f}"
            print(fila)
        
        #Activaciones FA
        print(f"\nFuncion de Activacion FA(D) = D^2 * log(D):")
        header = "    Patrón" + "".join(f"  FA(P,R{j+1})" for j in range(n_centros))
        print(header)
        for i in range(n_patrones):
            fila = f"    P{i+1:>3}"
            for j in range(n_centros):
                fila += f"   {FA[i,j]:8.4f}"
            print(fila)
        # Matriz A
        print(f"\nMatriz de Interpolacion A = [1 | FA] [{n_patrones} x ({n_centros} + 1)]:")
        print("    Patrón   bias" + " ".join(f"FA(P,R{j+1})" for j in range(n_centros)))
        print(header)
        for i in range(n_patrones):
            fila = f"    P{i+1:>3}  "
            fila += "  ".join(f"{A[i,k]:8.4f}" for k in range(A.shape[1]))
            print(fila)
        
        # Pesos
        print(f"\nSistema A·W = Yd -> W = pinv(A)·Yd:")
        print(f"Wo (bias) = {self.pesos[0, 0]:.6f}")
        for j in range(n_centros):
            print(f"    W{j+1}         = {self.pesos[j+1, 0]:.6f}")
        
        # Yr y EL
        Yr = A @ self.pesos
        EL = Yd_train - Yr
        print(f"\nSimulación sobre entrenamiento (Yr vs Yd):")
        print(f"    {'Patrón':>7}  {'Yr':>10}  {'Yd':>10}  {'EL = Yd-Yr':>12}")
        for i in range(n_patrones):
            print(f"    P{i+1:>3}    {Yr[i,0]:10.4f}  {Yd_train[i,0]:10.4f}  {EL[i,0]:12.4f}")

        EG = float(np.mean(np.abs(EL)))
        print(f"\n    EG = Σ|EL| / N = {EG:.6f}")
        print(sep)

    def _imprimir_detalle_simulacion(self, X, D, FA, A, Yr):
        n_centros = self.centros.shape[0]
        n_patrones = X.shape[0]
        sep = "─" * 60
        print(f"\n{sep}")
        print("  DETALLE DE SIMULACIÓN (modo verbose)")
        print(sep)
        print(f"\n▶ Distancias D y activaciones FA para {n_patrones} patrones:")
        for i in range(n_patrones):
            partes_d  = "  ".join(f"D{j+1}={D[i,j]:.4f}"  for j in range(n_centros))
            partes_fa = "  ".join(f"FA{j+1}={FA[i,j]:.4f}" for j in range(n_centros))
            print(f"    P{i+1:>3}: {partes_d}")
            print(f"          {partes_fa}")
            print(f"          Yr = {Yr[i,0]:.4f}")
        print(sep)


