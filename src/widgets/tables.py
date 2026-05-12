import tkinter as tk
from tkinter import ttk
import numpy as np
from styles import BG_CARD

def matriz_a_treeview(parent: tk.Frame, array: np.ndarray,
                        col_prefix: str = "C", row_prefix: str = "P",
                        max_rows: int = 200) -> ttk.Treeview:
    """Crea un Treeview con scroll que muestra un array 2-D."""
    n_rows, n_cols = array.shape
    col_ids = [f"{col_prefix}{j}" for j in range(n_cols)]
    cols = ("fila",) + tuple(col_ids)

    frame = tk.Frame(parent, bg=BG_CARD)
    frame.pack(fill="both", expand=True, padx=6, pady=4)

    vsb = ttk.Scrollbar(frame, orient="vertical")
    hsb = ttk.Scrollbar(frame, orient="horizontal")
    tv  = ttk.Treeview(frame, columns=cols, show="headings",
                        style="Dark.Treeview",
                        yscrollcommand=vsb.set,
                        xscrollcommand=hsb.set)
    vsb.config(command=tv.yview)
    hsb.config(command=tv.xview)

    # Encabezados
    tv.heading("fila", text="")
    tv.column("fila", width=52, anchor="e", stretch=False)
    for cid in col_ids:
        tv.heading(cid, text=cid)
        tv.column(cid, width=90, anchor="center")

    # Datos
    limite = min(n_rows, max_rows)
    for i in range(limite):
        valores = (f"{row_prefix}{i+1}",) + tuple(f"{v:.5f}" for v in array[i])
        tv.insert("", "end", values=valores)
    if n_rows > max_rows:
        tv.insert("", "end", values=(f"… {n_rows - max_rows} filas más …",))

    vsb.pack(side="right",  fill="y")
    hsb.pack(side="bottom", fill="x")
    tv.pack(fill="both", expand=True)
    return tv

def vector_a_treeview(parent: tk.Frame, array: np.ndarray,
                        row_labels: list, col_labels: list) -> ttk.Treeview:
    """Treeview para matrices de pesos (n_centros+1, n_salidas)."""
    cols = ("etiqueta",) + tuple(col_labels)
    frame = tk.Frame(parent, bg=BG_CARD)
    frame.pack(fill="both", expand=True, padx=6, pady=4)

    vsb = ttk.Scrollbar(frame, orient="vertical")
    tv  = ttk.Treeview(frame, columns=cols, show="headings",
                        style="Dark.Treeview", yscrollcommand=vsb.set)
    vsb.config(command=tv.yview)

    tv.heading("etiqueta", text="Peso")
    tv.column("etiqueta", width=90, anchor="w")
    for c in col_labels:
        tv.heading(c, text=c)
        tv.column(c, width=100, anchor="center")

    for i, lbl in enumerate(row_labels):
        fila = (lbl,) + tuple(f"{array[i, k]:.6f}" for k in range(array.shape[1]))
        tv.insert("", "end", values=fila)

    vsb.pack(side="right", fill="y")
    tv.pack(fill="both", expand=True)
    return tv


def tabla_particion(parent: tk.Frame, X: np.ndarray, Yd: np.ndarray,col_names: list, titulo: str):
    """Muestra X e Yd juntos en un Treeview."""
    data = np.hstack([X, Yd])
    n_col_x = X.shape[1]
    n_col_y = Yd.shape[1]
    all_cols = list(col_names[:n_col_x]) + [f"Yd{k}" for k in range(n_col_y)]
    matriz_a_treeview(parent, data,
                        col_prefix="", row_prefix="P")
