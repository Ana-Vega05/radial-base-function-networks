import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

def _predecir_clases(Yr: np.ndarray, Yd: np.ndarray) -> tuple:
    """
    Convierte salidas continuas de la red en etiquetas de clase.
    - Clasificación binaria (1 salida): umbral 0.5
    - Multiclase con codificación directa (1 salida): redondeo al entero más cercano y recorte al rango [0, max_clase]
    """
    n_clases_unicas = int(np.max(Yd)) + 1   # cantidad de clases presentes
    if Yr.shape[1] == 1:
        if n_clases_unicas == 2:
            # Clasificación binaria
            Yr_clase = (Yr.ravel() >= 0.5).astype(int)
        else:
            # Multiclase en una sola salida (codificación ordinal)
            Yr_clase = np.clip(np.round(Yr.ravel()).astype(int), 0, n_clases_unicas - 1)
    else:
        # Multiclase con one-hot → argmax
        Yr_clase = np.argmax(Yr, axis=1)

    Yd_clase = Yd.ravel().astype(int)
    return Yd_clase, Yr_clase
# Diagnóstico de Yr crudo (para investigar métricas perfectas)
def _diagnostico_yr(Yr: np.ndarray, Yd: np.ndarray, n_muestra: int = 10):
    """
    Imprime los valores crudos que sale de la red ANTES de aplicar
    el umbral o el redondeo.

    Esto permite verificar si la red realmente produce valores cerca de
    0/1 (aprendizaje genuino) o si hay algo raro (fuga de datos, etc.).
    """
    sep = "·" * 60
    print(f"\n{sep}")
    print("  DIAGNÓSTICO: Valores crudos Yr (antes del umbral/redondeo)")
    print(sep)

    n = Yr.shape[0]
    yr_flat = Yr.ravel()
    yd_flat = Yd.ravel()

    print(f"\n  Estadísticas de Yr crudo ({n} patrones):")
    print(f"    min   = {yr_flat.min():.6f}")
    print(f"    max   = {yr_flat.max():.6f}")
    print(f"    media = {yr_flat.mean():.6f}")
    print(f"    std   = {yr_flat.std():.6f}")

    # Histograma simple por rangos
    rangos = [(-np.inf, 0.0), (0.0, 0.1), (0.1, 0.4), (0.4, 0.6),
            (0.6, 0.9), (0.9, 1.0), (1.0, np.inf)]
    etiquetas = ["< 0.0", "0.0-0.1", "0.1-0.4", "0.4-0.6",
                "0.6-0.9", "0.9-1.0", "> 1.0"]
    print("\n  Distribución de Yr por rango:")
    for (lo, hi), etiq in zip(rangos, etiquetas):
        cnt = int(np.sum((yr_flat >= lo) & (yr_flat < hi)))
        barra = " " * (cnt * 30 // max(n, 1))
        print(f"    {etiq:>8}  {barra:<30}  {cnt:>4} ({cnt/n*100:.1f}%)")

    # Muestra de patrones individuales
    n_muestra = min(n_muestra, n)
    print(f"\n  Muestra de {n_muestra} patrones (Yd | Yr_crudo | diferencia):")
    print(f"  {'Patrón':>7}  {'Yd':>8}  {'Yr_crudo':>12}  {'|Yd-Yr|':>10}")
    for i in range(n_muestra):
        diff = abs(yd_flat[i] - yr_flat[i])
        print(f"  P{i+1:>5}    {yd_flat[i]:>8.4f}  {yr_flat[i]:>12.6f}  {diff:>10.6f}")

    # Verificar si hay patrones dudosos (Yr cerca del umbral 0.5)
    cerca_umbral = int(np.sum(np.abs(yr_flat - 0.5) < 0.1))
    if cerca_umbral > 0:
        print(f"\n  Patrones con Yr entre 0.4 y 0.6 (zona ambigua): {cerca_umbral}")
        print(f"    Estos pueden cambiar de clase si el modelo varía ligeramente.")
    else:
        print(f"\n  Ningún patrón cae en la zona ambigua [0.4, 0.6].")
        print(f"    Esto explica las métricas perfectas: la red separa limpiamente.")

    print(sep)
# Evaluación principal
def evaluar_modelo(modelo, X_test: np.ndarray, Yd_test: np.ndarray, verbose: bool = True, diagnostico: bool = True,) -> dict:
    """
    Evalúa el modelo entrenado sobre el conjunto de prueba.
    Retorna un dict con:
    - EG_test          : error general MAE
    - Yr               : salidas continuas de la red
    - EL               : errores locales (Yd - Yr)
    - Yd_clase         : etiquetas reales
    - Yr_clase         : etiquetas predichas
    - exactitud        : accuracy global
    - precision        : por clase (macro)
    - sensibilidad     : recall por clase (macro)
    - f1               : F1 score macro
    - confusion_matrix : matriz de confusión numpy
    - reporte          : texto del classification_report de sklearn
    """
    # ── Predicciones continuas ──
    Yr = modelo.predict(X_test)
    EL = Yd_test - Yr
    EG = float(np.mean(np.abs(EL)))

    # Diagnóstico de valores crudos (opcional, activo por defecto)
    if diagnostico:
        _diagnostico_yr(Yr, Yd_test)

    # ── Etiquetas de clase ──
    Yd_clase, Yr_clase = _predecir_clases(Yr, Yd_test)
    n_clases = len(np.unique(Yd_clase))
    promedio = "binary" if n_clases == 2 else "macro"

    # ── Métricas ──
    exactitud    = accuracy_score(Yd_clase, Yr_clase)
    precision    = precision_score(Yd_clase, Yr_clase, average=promedio, zero_division=0)
    sensibilidad = recall_score(Yd_clase, Yr_clase, average=promedio, zero_division=0)
    f1           = f1_score(Yd_clase, Yr_clase, average=promedio, zero_division=0)
    cm           = confusion_matrix(Yd_clase, Yr_clase)
    reporte      = classification_report(Yd_clase, Yr_clase, zero_division=0)

    resultados = {
        "EG_test": EG,
        "Yr": Yr,
        "EL": EL,
        "Yd_clase": Yd_clase,
        "Yr_clase": Yr_clase,
        "exactitud": exactitud,
        "precision": precision,
        "sensibilidad": sensibilidad,
        "f1": f1,
        "confusion_matrix": cm,
        "reporte": reporte,
    }

    if verbose:
        _imprimir_metricas(resultados, EG, n_clases)

    return resultados

def _imprimir_metricas(res: dict, EG: float, n_clases: int):
    sep = "═" * 60
    print(f"\n{sep}")
    print("  MÉTRICAS DE EVALUACIÓN (conjunto de prueba)")
    print(sep)
    print(f"  Error General (MAE)     : {EG:.6f}")
    print(f"  Exactitud (Accuracy)    : {res['exactitud']:.4f}  ({res['exactitud']*100:.2f}%)")
    print(f"  Precisión (Precision)   : {res['precision']:.4f}")
    print(f"  Sensibilidad (Recall)   : {res['sensibilidad']:.4f}")
    print(f"  F1-Score                : {res['f1']:.4f}")
    print(f"\n  Reporte completo por clase:")
    for linea in res["reporte"].splitlines():
        print(f"    {linea}")
    print(f"\n  Matriz de confusión ({n_clases}×{n_clases}):")
    cm = res["confusion_matrix"]
    for fila in cm:
        print("    " + "  ".join(f"{v:5d}" for v in fila))
    print(sep)


#  Gráficas
def graficar_resultados(resultados: dict, historial_EG: list,
                        historial_n_centros: list, error_optimo: float,
                        guardar: bool = False, prefijo: str = "rbf"):
    """
    Genera las 4 gráficas requeridas por el examen:

    1. YD vs YR          — salidas reales vs predichas (prueba)
    2. EG por iteración  — convergencia del entrenamiento
    3. |EL| por patrón   — error local absoluto en prueba
    4. Matriz de confusión — mapa de calor
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Red Neuronal RBF — Resultados", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    Yd_clase = resultados["Yd_clase"]
    Yr_clase = resultados["Yr_clase"]
    Yr       = resultados["Yr"].ravel()
    EL       = resultados["EL"].ravel()
    cm       = resultados["confusion_matrix"]
    n_test   = len(Yd_clase)

    #YD vs YR
    ax1 = fig.add_subplot(gs[0, 0])
    x_idx = np.arange(1, n_test + 1)
    ax1.plot(x_idx, Yd_clase, "o-", label="YD (deseada)", color="#1f77b4",
            linewidth=1.5, markersize=4)
    ax1.plot(x_idx, Yr_clase, "s--", label="YR (red)", color="#ff7f0e",
            linewidth=1.5, markersize=4)
    ax1.set_title("YD vs YR — Salidas deseadas y calculadas")
    ax1.set_xlabel("Patrón (prueba)")
    ax1.set_ylabel("Clase")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    #EG por iteración 
    ax2 = fig.add_subplot(gs[0, 1])
    iters = list(range(1, len(historial_EG) + 1))
    ax2.plot(historial_n_centros, historial_EG, "o-", color="#2ca02c",
            linewidth=2, markersize=6, label="EG por iteración")
    ax2.axhline(y=error_optimo, color="red", linestyle="--", linewidth=1.5,
                label=f"Error óptimo = {error_optimo}")
    ax2.set_title("Error General EG por iteración")
    ax2.set_xlabel("Número de centros (n_centros)")
    ax2.set_ylabel("EG")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    # Marcar convergencia si la hay
    for idx, eg in enumerate(historial_EG):
        if eg <= error_optimo:
            ax2.annotate("Convergencia", xy=(historial_n_centros[idx], eg),
                        xytext=(historial_n_centros[idx] + 0.5, eg + 0.01),
                        arrowprops=dict(arrowstyle="->", color="red"),
                        fontsize=8, color="red")
            break

    #|EL| por patrón
    ax3 = fig.add_subplot(gs[1, 0])
    abs_el = np.abs(EL)
    colores = ["#d62728" if v > error_optimo else "#1f77b4" for v in abs_el]
    ax3.bar(x_idx, abs_el, color=colores, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax3.axhline(y=error_optimo, color="red", linestyle="--", linewidth=1.5,
                label=f"Error óptimo = {error_optimo}")
    ax3.set_title("Error Local |EL| por patrón (prueba)")
    ax3.set_xlabel("Patrón (prueba)")
    ax3.set_ylabel("|EL|")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    #Matriz de confusión 
    ax4 = fig.add_subplot(gs[1, 1])
    n_clases = cm.shape[0]
    im = ax4.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax4, shrink=0.8)
    ax4.set_title("Matriz de confusión")
    ax4.set_xlabel("Clase predicha")
    ax4.set_ylabel("Clase real")
    ticks = np.arange(n_clases)
    ax4.set_xticks(ticks)
    ax4.set_yticks(ticks)
    ax4.set_xticklabels([f"C{i}" for i in range(n_clases)])
    ax4.set_yticklabels([f"C{i}" for i in range(n_clases)])
    umbral_color = cm.max() / 2
    for i in range(n_clases):
        for j in range(n_clases):
            color = "white" if cm[i, j] > umbral_color else "black"
            ax4.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    if guardar:
        ruta = f"{prefijo}_resultados.png"
        fig.savefig(ruta, dpi=150, bbox_inches="tight")
        print(f"\n  Gráficas guardadas en: {ruta}")

    plt.show()