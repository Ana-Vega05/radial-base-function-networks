import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from styles import BG_CARD, BG_DARK, ACCENT, WARNING, TEXT_MAIN, TEXT_DIM, BORDER

class PanelHistogramas(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG_CARD, **kwargs)
        self._canvas = None
        self._build()

    def _build(self):
        self._holder = tk.Frame(self, bg=BG_CARD)
        self._holder.pack(fill="both", expand=True, padx=8, pady=4)
        self._placeholder = tk.Label(self._holder, text="Cargue un dataset para ver los histogramas.",bg=BG_CARD, fg=TEXT_DIM, font=("Segoe UI", 9))
        self._placeholder.pack(expand=True)

    def actualizar(self, X: np.ndarray, nombres: list):
        if self._canvas:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        self._placeholder.pack_forget()

        n = X.shape[1]
        ncols = min(n, 4)
        nrows = (n + ncols - 1) // ncols

        figw = max(8, ncols * 2.6)
        figh = max(2.2, nrows * 2.0)
        fig = Figure(figsize=(figw, figh), facecolor=BG_CARD)

        for i in range(n):
            ax = fig.add_subplot(nrows, ncols, i + 1)
            data_col = X[:, i]
            ax.hist(data_col, bins=20, color=ACCENT, edgecolor=BG_DARK,
                    alpha=0.85, linewidth=0.5)
            ax.axvline(np.mean(data_col), color=WARNING, linewidth=1.5,linestyle="--", label="Media")
            ax.set_title(nombres[i], color=TEXT_MAIN, fontsize=7.5, pad=3)
            ax.set_facecolor(BG_DARK)
            ax.tick_params(colors=TEXT_DIM, labelsize=6)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.set_xlabel(f"μ={np.mean(data_col):.3f}  σ={np.std(data_col):.3f}",color=TEXT_DIM, fontsize=6)

        fig.tight_layout(pad=1.0)
        self._canvas = FigureCanvasTkAgg(fig, master=self._holder)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True)