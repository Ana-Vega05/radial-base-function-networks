import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from styles import BG_CARD, BG_DARK, ACCENT, ACCENT2, SUCCESS, WARNING, DANGER, TEXT_MAIN, TEXT_DIM, BORDER, FONT_SMALL

class PanelDistribucionClases(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG_CARD, **kwargs)
        self._canvas = None
        self._build()

    def _build(self):
        self._holder = tk.Frame(self, bg=BG_CARD)
        self._holder.pack(fill="both", expand=True, padx=8, pady=4)
        self._placeholder = tk.Label(self._holder, text="Sin datos.",bg=BG_CARD, fg=TEXT_DIM, font=FONT_SMALL)
        self._placeholder.pack(expand=True)

    def actualizar(self, Yd_clase: np.ndarray, n_clases: int):
        if self._canvas:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        self._placeholder.pack_forget()

        clases, conteos = np.unique(Yd_clase, return_counts=True)
        total = len(Yd_clase)
        colores = [ACCENT, ACCENT2, SUCCESS, WARNING, DANGER,"#b48ead", "#ebcb8b", "#88c0d0"]

        fig = Figure(figsize=(3.2, 3.0), facecolor=BG_CARD)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG_DARK)

        bars = ax.bar(
            [f"C{c}" for c in clases], conteos,
            color=[colores[i % len(colores)] for i in range(len(clases))],
            edgecolor=BG_DARK, linewidth=0.5, width=0.6)

        for bar, cnt in zip(bars, conteos):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(conteos) * 0.02,
                    f"{cnt}\n({cnt/total*100:.1f}%)",
                    ha="center", va="bottom",
                    color=TEXT_MAIN, fontsize=7, fontweight="bold")

        ax.set_title("Clases", color=TEXT_MAIN, fontsize=9)
        ax.set_xlabel("Clase", color=TEXT_DIM, fontsize=8)
        ax.set_ylabel("Patrones", color=TEXT_DIM, fontsize=8)
        ax.tick_params(colors=TEXT_DIM, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

        fig.tight_layout(pad=1.2)
        self._canvas = FigureCanvasTkAgg(fig, master=self._holder)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True)