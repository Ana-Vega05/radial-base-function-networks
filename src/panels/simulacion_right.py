# -*- coding: utf-8 -*-
"""Panel derecho de Simulación – predicción, gráfico de barras y Yr crudo."""

import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from styles import (
    BG_CARD, BG_DARK, ACCENT, ACCENT2, TEXT_MAIN, TEXT_DIM,
    SUCCESS, WARNING, BORDER, FONT_SMALL
)
from widgets import section_label, separator


class PanelSimulacionRight(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG_CARD, **kwargs)
        self._build()

    def _build(self):
        # Predicción principal
        section_label(self, "Predicción")
        separator(self)

        res_outer = tk.Frame(self, bg=BG_DARK, height=70)
        res_outer.pack(fill="x", padx=12, pady=6)
        res_outer.pack_propagate(False)

        self.pred_lbl = tk.Label(res_outer, text="—", bg=BG_DARK, fg=ACCENT2,font=("Segoe UI", 30, "bold"))
        self.pred_lbl.pack(side="left", padx=20, pady=10)

        self.conf_lbl = tk.Label(res_outer, text="", bg=BG_DARK, fg=TEXT_DIM,font=("Segoe UI", 12))
        self.conf_lbl.pack(side="left", pady=10)

        self.real_lbl = tk.Label(res_outer, text="", bg=BG_DARK, fg=SUCCESS,font=("Segoe UI", 11))
        self.real_lbl.pack(side="right", padx=20)

        separator(self)
        section_label(self, "Scores por Clase (Yr crudo)")
        separator(self)

        # Gráfico de barras
        self._build_bar()

        separator(self)
        section_label(self, "Detalle Yr (valores crudos de red)")
        separator(self)

        # Marco para la tabla Yr
        self._yr_frame = tk.Frame(self, bg=BG_CARD, height=90)
        self._yr_frame.pack(fill="x", padx=12, pady=4)
        self._show_yr_placeholder()

    def _build_bar(self):
        self.fig_bar = Figure(figsize=(5.5, 2.4), facecolor=BG_CARD)
        self.ax_bar = self.fig_bar.add_subplot(111)
        self.ax_bar.set_facecolor(BG_DARK)
        # Barras iniciales vacías
        self.ax_bar.tick_params(colors=TEXT_DIM, labelsize=8)
        for spine in self.ax_bar.spines.values():
            spine.set_edgecolor(BORDER)
        self.fig_bar.tight_layout(pad=1.2)
        self.canvas_bar = FigureCanvasTkAgg(self.fig_bar, master=self)
        self.canvas_bar.draw()
        self.canvas_bar.get_tk_widget().pack(fill="x", padx=12, pady=4)

    # Métodos públicos

    def set_prediction(self, predicted, confidence):
        self.pred_lbl.configure(text=str(predicted), fg=ACCENT2)
        self.conf_lbl.configure(text=f"  Confianza: {confidence:.1%}", fg=TEXT_DIM)

    def set_real_class(self, clase):
        self.real_lbl.configure(text=f"Clase real: {clase}", fg=SUCCESS)

    def clear_real_class(self):
        self.real_lbl.configure(text="")

    def update_bar_chart(self, clases, scores, idx_pred):
        self.ax_bar.clear()
        self.ax_bar.set_facecolor(BG_DARK)
        colores = [SUCCESS if i == idx_pred else ACCENT for i in range(len(clases))]
        bars = self.ax_bar.barh(clases, scores, color=colores, height=0.5, alpha=0.9)

        for bar, v in zip(bars, scores):
            self.ax_bar.text(
                min(v + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", color=TEXT_MAIN, fontsize=8)

        self.ax_bar.set_xlim(0, 1.0)
        self.ax_bar.set_xlabel("Score / Probabilidad", color=TEXT_DIM, fontsize=9)
        self.ax_bar.tick_params(colors=TEXT_DIM, labelsize=8)
        for spine in self.ax_bar.spines.values():
            spine.set_edgecolor(BORDER)
        self.fig_bar.tight_layout(pad=1.2)
        self.canvas_bar.draw()

    def show_yr_raw(self, yr_row: np.ndarray, clases: list):
        """Muestra una tabla con los valores crudos de Yr."""
        for w in self._yr_frame.winfo_children():
            w.destroy()

        cols = ("clase", "yr_crudo")
        tv = ttk.Treeview(self._yr_frame, columns=cols, show="headings",style="Dark.Treeview", height=min(len(clases), 5))
        tv.heading("clase", text="Clase")
        tv.heading("yr_crudo", text="Yr (crudo)")
        tv.column("clase", width=130, anchor="center")
        tv.column("yr_crudo", width=140, anchor="center")

        for c, v in zip(clases, yr_row):
            tv.insert("", "end", values=(c, f"{v:.8f}"))

        tv.pack(fill="x", padx=4, pady=4)

    def clear(self):
        self.pred_lbl.configure(text="—", fg=ACCENT2)
        self.conf_lbl.configure(text="")
        self.real_lbl.configure(text="")
        self.ax_bar.clear()
        self.ax_bar.set_facecolor(BG_DARK)
        self.ax_bar.tick_params(colors=TEXT_DIM)
        self.fig_bar.tight_layout(pad=1.2)
        self.canvas_bar.draw()
        for w in self._yr_frame.winfo_children():
            w.destroy()
        self._show_yr_placeholder()

    def _show_yr_placeholder(self):
        tk.Label(self._yr_frame, text="Sin simulación aún.",bg=BG_CARD, fg=TEXT_DIM, font=FONT_SMALL).pack(pady=8)