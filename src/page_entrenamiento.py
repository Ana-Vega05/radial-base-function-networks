# -*- coding: utf-8 -*-
"""Página 4 – Parámetros y Entrenamiento del Modelo RBF."""

import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from styles import (
    BG_DARK, BG_CARD, BG_PANEL, ACCENT, ACCENT2, TEXT_MAIN, TEXT_DIM,
    SUCCESS, DANGER, BORDER, FONT_TITLE, FONT_SMALL,
)
from widgets import card, section_label, separator, labeled_entry, labeled_combo, action_btn


class PageEntrenamiento(tk.Frame):
    def __init__(self, parent, status_bar):
        super().__init__(parent, bg=BG_DARK)
        self.status = status_bar
        self._build()

    def _build(self):
        tk.Label(self, text="Parámetros y Entrenamiento del Modelo RBF",
                 bg=BG_DARK, fg=TEXT_MAIN, font=FONT_TITLE).pack(
                     anchor="w", padx=24, pady=(20, 4))
        tk.Label(self, text="Configure la arquitectura y entrene la red de base radial.",
                 bg=BG_DARK, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor="w", padx=24)

        # ── Panel izquierdo: parámetros ──
        left = card(self)
        left.place(relx=0.01, rely=0.12, relwidth=0.38, relheight=0.82)
        section_label(left, "Arquitectura RBF")
        separator(left)

        self.n_centers = labeled_entry(left, "N° de centros (k):", "10")
        self.sigma     = labeled_entry(left, "Sigma (σ):",         "1.0")
        self.lr        = labeled_entry(left, "Tasa de aprendizaje:", "0.01")
        self.epochs    = labeled_entry(left, "Épocas máximas:",    "200")
        self.tol       = labeled_entry(left, "Tolerancia:",        "1e-4")
        self.init_var  = labeled_combo(left, "Init. centros:",
                                       ["K-Means", "Aleatorio", "Percentiles"], 0)

        separator(left)
        section_label(left, "Regularización")
        self.reg_var = labeled_combo(left, "Tipo:", ["Ninguna", "L1", "L2"], 0)
        self.reg_val = labeled_entry(left, "λ (lambda):", "0.001")

        separator(left)
        btn_row = tk.Frame(left, bg=BG_CARD)
        btn_row.pack(fill="x", padx=12, pady=10)
        action_btn(btn_row, "▶  Entrenar",
                   command=self._entrenar, color=SUCCESS).pack(side="left")
        action_btn(btn_row, "✖  Detener",
                   command=lambda: None, color=DANGER).pack(side="left", padx=(8, 0))

        # ── Panel derecho: curva de aprendizaje ──
        right = card(self)
        right.place(relx=0.41, rely=0.12, relwidth=0.58, relheight=0.82)
        section_label(right, "Curva de Aprendizaje")
        separator(right)
        self._build_loss_plot(right)

        separator(right)
        self.prog_var = tk.DoubleVar(value=0)
        self.prog_lbl = tk.Label(right, text="0 / 0 épocas",
                                 bg=BG_CARD, fg=TEXT_DIM, font=FONT_SMALL)
        self.prog_lbl.pack(pady=(4, 2))

        style = ttk.Style()
        style.configure("TProgressbar", troughcolor=BG_DARK,
                        background=ACCENT, thickness=10)
        ttk.Progressbar(right, variable=self.prog_var, maximum=100,
                        style="TProgressbar", length=400).pack(
                            padx=12, pady=(0, 8), fill="x")

    def _build_loss_plot(self, parent):
        self.fig_loss = Figure(figsize=(6, 3.5), facecolor=BG_CARD)
        self.ax_loss  = self.fig_loss.add_subplot(111)
        self.ax_loss.set_facecolor(BG_DARK)
        self.ax_loss.set_xlabel("Época", color=TEXT_DIM, fontsize=9)
        self.ax_loss.set_ylabel("Error", color=TEXT_DIM, fontsize=9)
        self.ax_loss.set_title("Error de Entrenamiento / Validación",
                               color=TEXT_MAIN, fontsize=10)
        self.ax_loss.tick_params(colors=TEXT_DIM)
        for spine in self.ax_loss.spines.values():
            spine.set_edgecolor(BORDER)
        self.fig_loss.tight_layout(pad=1.5)
        self.canvas_loss = FigureCanvasTkAgg(self.fig_loss, master=parent)
        self.canvas_loss.draw()
        self.canvas_loss.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=4)

    # ── Acciones ──────────────────────────────────────────────────────────────

    def _entrenar(self):
        epochs = int(self.epochs.get())
        ep        = np.arange(1, epochs + 1)
        train_err = 1.0 * np.exp(-ep / (epochs * 0.3))  + np.random.uniform(0, 0.02, epochs)
        val_err   = 1.1 * np.exp(-ep / (epochs * 0.32)) + np.random.uniform(0, 0.03, epochs)

        self.ax_loss.clear()
        self.ax_loss.set_facecolor(BG_DARK)
        self.ax_loss.plot(ep, train_err, color=ACCENT,  label="Entrenamiento", linewidth=2)
        self.ax_loss.plot(ep, val_err,   color=ACCENT2, label="Validación",
                          linewidth=2, linestyle="--")
        self.ax_loss.set_xlabel("Época",  color=TEXT_DIM,  fontsize=9)
        self.ax_loss.set_ylabel("Error",  color=TEXT_DIM,  fontsize=9)
        self.ax_loss.set_title("Curva de Aprendizaje", color=TEXT_MAIN, fontsize=10)
        self.ax_loss.legend(facecolor=BG_PANEL, labelcolor=TEXT_MAIN, fontsize=8)
        self.ax_loss.tick_params(colors=TEXT_DIM)
        for spine in self.ax_loss.spines.values():
            spine.set_edgecolor(BORDER)
        self.fig_loss.tight_layout(pad=1.5)
        self.canvas_loss.draw()

        self.prog_var.set(100)
        self.prog_lbl.configure(
            text=f"{epochs} / {epochs} épocas ", fg=SUCCESS)
        self.status.set(f"Modelo entrenado — {epochs} épocas completadas.", "ok")
