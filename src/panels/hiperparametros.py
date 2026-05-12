# -*- coding: utf-8 -*-
"""Panel izquierdo – Hiperparámetros y controles de entrenamiento."""

import tkinter as tk
from tkinter import ttk

from styles import (
    BG_CARD, ACCENT, ACCENT2, TEXT_DIM, SUCCESS, DANGER, FONT_SMALL,
)
from widgets import danger_btn, primary_btn, section_label, separator, labeled_entry, labeled_combo, action_btn, success_btn


class PanelHiperparametros(tk.Frame):
    """
    Contiene los campos de hiperparámetros, barra de progreso,
    indicadores de estado y botones Entrenar/Detener.
    """

    def __init__(self, parent, app_state: dict,
                 on_entrenar, on_detener, **kwargs):
        super().__init__(parent, bg=BG_CARD, **kwargs)
        self.app_state = app_state
        self._on_entrenar = on_entrenar
        self._on_detener = on_detener

        self._build()

    # Construccion de la interfaz

    def _build(self):
        section_label(self, "Hiperparámetros")
        separator(self)

        self.n_centers = labeled_entry(self, "N° de centros (k):", "10")
        self.lr        = labeled_entry(self, "Error óptimo:",       "0.05")
        self.epochs    = labeled_entry(self, "Iteraciones máx.:",   "50")
        self.part_var  = labeled_combo(self, "Partición:",["80-10-10", "70-15-15"], 0)
        self.seed_var  = labeled_combo(self, "Semilla:",["Aleatorio", "Fija (42)"], 0)

        separator(self)
        section_label(self, "Estado")

        self.prog_var = tk.DoubleVar(value=0)
        self.prog_lbl = tk.Label(self, text="Sin entrenar",bg=BG_CARD, fg=TEXT_DIM, font=FONT_SMALL)
        self.prog_lbl.pack(pady=(4, 2))

        ttk.Progressbar(self, variable=self.prog_var, maximum=100,
                        style="TProgressbar").pack(padx=12, pady=(0, 6), fill="x")

        self.eg_lbl = tk.Label(self, text="EG: —",bg=BG_CARD, fg=ACCENT2, font=("Segoe UI", 11, "bold"))
        self.eg_lbl.pack(pady=2)

        self.conv_lbl = tk.Label(self, text="",bg=BG_CARD, fg=TEXT_DIM, font=FONT_SMALL)
        self.conv_lbl.pack(pady=2)

        separator(self)

        btn_row = tk.Frame(self, bg=BG_CARD)
        btn_row.pack(fill="x", padx=12, pady=10)
        success_btn(btn_row, "▶  Entrenar", self._on_entrenar).pack(side="left")
        danger_btn(btn_row, "■  Detener", self._on_detener).pack(side="left", padx=(8, 0))

        # Botón de guardar (inicia deshabilitado)
        self.btn_guardar = success_btn(btn_row, "💾 Guardar modelo", lambda: None)
        self.btn_guardar.pack(side="left", padx=(8, 0))
        self.btn_guardar.configure(state="disabled")

    #MEtodos pUblicos para la pagina principal

    def cargar_desde_config(self):
        """Carga los valores iniciales desde app_state['config']."""
        cfg = self.app_state.get("config")
        if cfg:
            self.n_centers.set(str(cfg.n_centros))
            self.lr.set(str(cfg.error_optimo))
            self.epochs.set(str(cfg.max_iteraciones))

    def obtener_valores(self):
        """Retorna los valores actuales del panel como un diccionario."""
        return {
            "n_centros": self.n_centers.get(),
            "error_optimo": self.lr.get(),
            "max_iteraciones": self.epochs.get(),
            "particion": self.part_var.get(),
            "semilla": self.seed_var.get(),
        }
    
    def habilitar_guardado(self, comando):
        self.btn_guardar.configure(state="normal", command=comando)

    def deshabilitar_guardado(self):
        self.btn_guardar.configure(state="disabled", command=None)

    def actualizar_progreso(self, iter_n, total, eg, pct):
        """Actualiza la barra de progreso y las etiquetas de estado."""
        self.prog_var.set(pct)
        self.prog_lbl.configure(
            text=f"Iteración {iter_n} / {total}  —  EG = {eg:.5f}")
        self.eg_lbl.configure(text=f"EG: {eg:.6f}")

    def entrenamiento_terminado(self, hist_EG, convergio):
        """Actualiza los indicadores al finalizar el entrenamiento."""
        self.prog_var.set(100)
        if convergio:
            msg = f"CONVERGE — EG final: {hist_EG[-1]:.6f}"
            color = SUCCESS
        else:
            msg = f"NO CONVERGE — mejor EG: {min(hist_EG):.6f}"
            color = DANGER

        self.prog_lbl.configure(
            text=f"{len(hist_EG)} iteraciones completadas", fg=color)
        self.conv_lbl.configure(text=msg, fg=color)
        self.eg_lbl.configure(text=f"EG: {hist_EG[-1]:.6f}", fg=color)

    def resetear_estado(self):
        """Reinicia la UI antes de un nuevo entrenamiento."""
        self.prog_var.set(0)
        self.prog_lbl.configure(text="Entrenando…", fg=ACCENT)
        self.conv_lbl.configure(text="", fg=TEXT_DIM)
        self.eg_lbl.configure(text="EG: …")