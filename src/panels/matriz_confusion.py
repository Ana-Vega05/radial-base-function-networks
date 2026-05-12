import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from styles import BG_CARD, BG_DARK, TEXT_MAIN, TEXT_DIM, BORDER

class PanelMatrizConfusion(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG_CARD, **kwargs)
        self._canvas = None
        self._build()

    def _build(self):
        self._holder = tk.Frame(self, bg=BG_CARD)
        self._holder.pack(fill="both", expand=True, padx=8, pady=4)
        self._placeholder = tk.Label(self._holder,
                                     text="Ejecute el pipeline para ver la matriz.",
                                     bg=BG_CARD, fg=TEXT_DIM, font=("Segoe UI", 9))
        self._placeholder.pack(expand=True)

    def actualizar(self, cm: np.ndarray):
        if self._canvas:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        self._placeholder.pack_forget()

        n_clases = cm.shape[0]
        fig_w = max(3.8, n_clases * 1.2)
        fig_h = max(3.0, n_clases * 1.0)
        fig = Figure(figsize=(fig_w, fig_h), facecolor=BG_CARD)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG_DARK)

        im = ax.imshow(cm, cmap="Blues", aspect="auto", vmin=0, vmax=cm.max())
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.ax.tick_params(colors=TEXT_DIM, labelsize=7)

        labels = [f"C{i}" for i in range(n_clases)]
        ax.set_xticks(range(n_clases))
        ax.set_yticks(range(n_clases))
        ax.set_xticklabels(labels, color=TEXT_DIM, fontsize=9)
        ax.set_yticklabels(labels, color=TEXT_DIM, fontsize=9)
        ax.set_xlabel("Clase Predicha", color=TEXT_DIM, fontsize=9)
        ax.set_ylabel("Clase Real",     color=TEXT_DIM, fontsize=9)
        ax.set_title("Matriz de Confusión", color=TEXT_MAIN, fontsize=10)

        umbral = cm.max() / 2.0
        for i in range(n_clases):
            for j in range(n_clases):
                color = "white" if cm[i, j] > umbral else "black"
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")

        fig.tight_layout(pad=1.2)
        self._canvas = FigureCanvasTkAgg(fig, master=self._holder)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True)