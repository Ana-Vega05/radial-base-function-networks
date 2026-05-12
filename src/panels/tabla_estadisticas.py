import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.stats import skew, kurtosis
from styles import BG_CARD, TEXT_DIM, WARNING

class PanelTablaEstadisticas(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG_CARD, **kwargs)
        self._build()

    def _build(self):
        cols = ("Variable", "Media", "Desv. Std", "Mín", "Mediana",
                "Máx", "Rango", "Asimetría", "Curtosis")
        frame = tk.Frame(self, bg=BG_CARD)
        frame.pack(fill="both", expand=True, padx=8, pady=4)

        vsb = ttk.Scrollbar(frame, orient="vertical")
        hsb = ttk.Scrollbar(frame, orient="horizontal")
        self.tree = ttk.Treeview(
            frame, columns=cols, show="headings", style="Dark.Treeview",
            yscrollcommand=vsb.set, xscrollcommand=hsb.set, height=7)
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)

        widths = {"Variable": 130, "Media": 90, "Desv. Std": 90,"Mín": 80, "Mediana": 85, "Máx": 80,"Rango": 80, "Asimetría": 85, "Curtosis": 80}
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=widths.get(c, 85), anchor="center")
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

    def actualizar(self, X, Yd, input_cols, target_col):
        for row in self.tree.get_children():
            self.tree.delete(row)

        Yd_clase = Yd.ravel().astype(int) if Yd.shape[1] == 1 else np.argmax(Yd, axis=1)
        all_data  = np.hstack([X, Yd_clase.reshape(-1, 1)])
        all_names = input_cols + [target_col]

        for i, nombre in enumerate(all_names):
            d = all_data[:, i].astype(float)
            try:
                asim = skew(d)
                kurt = kurtosis(d)
            except Exception:
                asim, kurt = float("nan"), float("nan")

            tag = "target" if nombre == target_col else ""
            self.tree.insert("", "end", tags=(tag,), values=(
                nombre,
                f"{np.mean(d):.4f}",
                f"{np.std(d, ddof=1):.4f}",
                f"{np.min(d):.4f}",
                f"{np.median(d):.4f}",
                f"{np.max(d):.4f}",
                f"{np.max(d) - np.min(d):.4f}",
                f"{asim:.4f}",
                f"{kurt:.4f}",
            ))
        self.tree.tag_configure("target", foreground=WARNING)