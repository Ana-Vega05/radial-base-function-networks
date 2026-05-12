import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from styles import BG_CARD, BG_DARK, BG_PANEL, ACCENT, ACCENT2, SUCCESS,TEXT_MAIN, TEXT_DIM, DANGER, WARNING, BORDER, FONT_SMALL

class PanelGraficasEvaluacion(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG_CARD, **kwargs)
        self._canvas = None
        self._build()

    def _build(self):
        self._holder = tk.Frame(self, bg=BG_CARD)
        self._holder.pack(fill="both", expand=True, padx=8, pady=4)
        self._placeholder = tk.Label(self._holder,text="Ejecute el pipeline para ver las gráficas.",bg=BG_CARD, fg=TEXT_DIM, font=FONT_SMALL)
        self._placeholder.pack(expand=True)

    def actualizar(self, res: dict, hist_eg: list, hist_cen: list, error_optimo: float):
        if self._canvas:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        self._placeholder.pack_forget()

        Yd_clase = res["Yd_clase"]
        Yr_clase = res["Yr_clase"]
        EL_raw   = res["EL"]
        if EL_raw.ndim == 2 and EL_raw.shape[1] > 1:
            abs_el = np.linalg.norm(EL_raw, axis=1)
        else:
            abs_el = np.abs(EL_raw.ravel())

        n_test = len(Yd_clase)
        x_idx  = np.arange(1, n_test + 1)

        fig = Figure(figsize=(13, 3.0), facecolor=BG_CARD)

        # Gráfica 1: YD vs YR
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_facecolor(BG_DARK)
        ax1.plot(x_idx, Yd_clase, "o-", color=ACCENT,label="YD (real)", linewidth=1.5, markersize=3)
        ax1.plot(x_idx, Yr_clase, "s--", color=ACCENT2,label="YR (predicho)", linewidth=1.5, markersize=3)
        ax1.set_title("YD vs YR — Salidas", color=TEXT_MAIN, fontsize=9)
        ax1.set_xlabel("Patrón (prueba)", color=TEXT_DIM, fontsize=8)
        ax1.set_ylabel("Clase", color=TEXT_DIM, fontsize=8)
        ax1.legend(facecolor=BG_PANEL, labelcolor=TEXT_MAIN, fontsize=7)
        ax1.tick_params(colors=TEXT_DIM, labelsize=7)
        for sp in ax1.spines.values(): sp.set_edgecolor(BORDER)

        # Gráfica 2: EG por iteración
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_facecolor(BG_DARK)
        if hist_eg:
            iters = list(range(1, len(hist_eg) + 1))
            ax2.plot(iters, hist_eg, "o-", color=SUCCESS,linewidth=2, markersize=5, label="EG")
            ax2.axhline(y=error_optimo, color=DANGER, linestyle="--",
                        linewidth=1.5, label=f"Óptimo = {error_optimo}")
            for i, eg in enumerate(hist_eg):
                if eg <= error_optimo:
                    ax2.plot(i + 1, eg, "D", color=WARNING,markersize=9, zorder=5)
                    break
            ax2.legend(facecolor=BG_PANEL, labelcolor=TEXT_MAIN, fontsize=7)
        ax2.set_title("EG por Iteración", color=TEXT_MAIN, fontsize=9)
        ax2.set_xlabel("Iteración", color=TEXT_DIM, fontsize=8)
        ax2.set_ylabel("EG", color=TEXT_DIM, fontsize=8)
        ax2.tick_params(colors=TEXT_DIM, labelsize=7)
        for sp in ax2.spines.values(): sp.set_edgecolor(BORDER)

        # Gráfica 3: |EL| por patrón
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_facecolor(BG_DARK)
        colores_bar = [DANGER if v > error_optimo else ACCENT for v in abs_el]
        ax3.bar(x_idx, abs_el, color=colores_bar,
                alpha=0.85, edgecolor=BG_DARK, linewidth=0.3)
        if error_optimo > 0:
            ax3.axhline(y=error_optimo, color=DANGER, linestyle="--",
                        linewidth=1.5, label=f"Óptimo = {error_optimo}")
            ax3.legend(facecolor=BG_PANEL, labelcolor=TEXT_MAIN, fontsize=7)
        n_sobre = int(np.sum(abs_el > error_optimo))
        ax3.set_title(f"|EL| por Patrón  ({n_sobre}/{n_test} sobre umbral)",color=TEXT_MAIN, fontsize=8.5)
        ax3.set_xlabel("Patrón (prueba)", color=TEXT_DIM, fontsize=8)
        ax3.set_ylabel("|EL|", color=TEXT_DIM, fontsize=8)
        ax3.tick_params(colors=TEXT_DIM, labelsize=7)
        for sp in ax3.spines.values(): sp.set_edgecolor(BORDER)

        fig.tight_layout(pad=1.3)
        self._canvas = FigureCanvasTkAgg(fig, master=self._holder)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True)