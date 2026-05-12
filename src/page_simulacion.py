# -*- coding: utf-8 -*-
"""Página 5 – Simulación del Modelo."""

import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from styles import (
    BG_DARK, BG_CARD, ACCENT, ACCENT2, TEXT_MAIN, TEXT_DIM, SUCCESS, BORDER,
    FONT_TITLE, FONT_SMALL,
)
from widgets import card, section_label, separator, labeled_entry, labeled_combo, action_btn


class PageSimulacion(tk.Frame):
    def __init__(self, parent, status_bar):
        super().__init__(parent, bg=BG_DARK)
        self.status = status_bar
        self._build()

    def _build(self):
        tk.Label(self, text="Simulación del Modelo", bg=BG_DARK, fg=TEXT_MAIN,
                 font=FONT_TITLE).pack(anchor="w", padx=24, pady=(20, 4))
        tk.Label(self,
                 text="Ingrese valores de entrada para obtener una predicción del modelo RBF.",
                 bg=BG_DARK, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor="w", padx=24)

        # ── Panel izquierdo: entradas ──
        left = card(self)
        left.place(relx=0.01, rely=0.12, relwidth=0.38, relheight=0.82)
        section_label(left, "Valores de Entrada")
        separator(left)

        self.sim_entries: dict = {}
        for f in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
            self.sim_entries[f] = labeled_entry(left, f"{f}:", "0.0")

        separator(left)
        self.mode_var = labeled_combo(
            left, "Fuente de datos:",
            ["Entrada manual", "Conjunto de prueba", "Cargar CSV"], 0)
        separator(left)

        btn_row = tk.Frame(left, bg=BG_CARD)
        btn_row.pack(fill="x", padx=12, pady=10)
        action_btn(btn_row, "▶  Simular",
                   command=self._simular, color=SUCCESS).pack(side="left")
        action_btn(btn_row, "Limpiar",
                   command=self._limpiar, color=BG_DARK).pack(side="left", padx=(8, 0))

        # ── Panel derecho: resultado ──
        right = card(self)
        right.place(relx=0.41, rely=0.12, relwidth=0.58, relheight=0.82)
        section_label(right, "Resultado de la Predicción")
        separator(right)

        res_box = tk.Frame(right, bg=BG_DARK, height=80)
        res_box.pack(fill="x", padx=12, pady=8)
        self.pred_lbl = tk.Label(res_box, text="—", bg=BG_DARK, fg=ACCENT2,
                                 font=("Segoe UI", 28, "bold"))
        self.pred_lbl.pack(pady=16)

        separator(right)
        section_label(right, "Probabilidades por Clase")
        separator(right)
        self._build_bar(right)

    def _build_bar(self, parent):
        self.fig_bar = Figure(figsize=(5, 3), facecolor=BG_CARD)
        self.ax_bar  = self.fig_bar.add_subplot(111)
        self.ax_bar.set_facecolor(BG_DARK)
        classes = ["Setosa", "Versicolor", "Virginica"]
        self.bars = self.ax_bar.barh(classes, [0.0, 0.0, 0.0],
                                     color=ACCENT, height=0.5)
        self.ax_bar.set_xlim(0, 1)
        self.ax_bar.set_xlabel("Probabilidad", color=TEXT_DIM, fontsize=9)
        self.ax_bar.tick_params(colors=TEXT_DIM)
        for spine in self.ax_bar.spines.values():
            spine.set_edgecolor(BORDER)
        self.fig_bar.tight_layout(pad=1.5)
        self.canvas_bar = FigureCanvasTkAgg(self.fig_bar, master=parent)
        self.canvas_bar.draw()
        self.canvas_bar.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=4)

    # ── Acciones ──────────────────────────────────────────────────────────────

    def _simular(self):
        probs     = np.random.dirichlet([3, 1, 1])
        classes   = ["Setosa", "Versicolor", "Virginica"]
        predicted = classes[int(np.argmax(probs))]
        self.pred_lbl.configure(text=predicted)
        for bar, p in zip(self.bars, probs):
            bar.set_width(p)
        for bar, p in zip(self.bars, probs):
            bar.set_color(SUCCESS if p == max(probs) else ACCENT)
        self.canvas_bar.draw()
        self.status.set(
            f"Predicción: {predicted}  (confianza: {max(probs):.1%})", "ok")

    def _limpiar(self):
        for var in self.sim_entries.values():
            var.set("0.0")
        self.pred_lbl.configure(text="—")
        for bar in self.bars:
            bar.set_width(0)
        self.canvas_bar.draw()
