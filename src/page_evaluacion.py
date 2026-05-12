# -*- coding: utf-8 -*-
"""Vista de Evaluación — métricas reales, matriz de confusión y gráficas del pipeline."""

import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from styles import (
    BG_DARK, BG_CARD, BG_PANEL, ACCENT, ACCENT2, TEXT_MAIN, TEXT_DIM,
    SUCCESS, WARNING, DANGER, BORDER, FONT_TITLE, FONT_SMALL,
)
from widgets import card, section_label, separator, action_btn


class PageEvaluacion(tk.Frame):
    def __init__(self, parent, status_bar, app_state: dict):
        super().__init__(parent, bg=BG_DARK)
        self.status    = status_bar
        self.app_state = app_state
        self._canvas_cm     = None
        self._canvas_charts = None
        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        tk.Label(self, text="Evaluación del Modelo", bg=BG_DARK,
                 fg=TEXT_MAIN, font=FONT_TITLE).pack(
                     anchor="w", padx=24, pady=(20, 4))
        tk.Label(self,
                 text="Métricas reales, matriz de confusión y gráficas de rendimiento del pipeline.",
                 bg=BG_DARK, fg=TEXT_DIM, font=FONT_SMALL).pack(
                     anchor="w", padx=24)

        # Botón calcular (arriba a la derecha)
        action_btn(self, "Actualizar Evaluación",
                   command=self._calcular, color=ACCENT).place(relx=0.75, rely=0.04)

        self.lbl_estado = tk.Label(self, text="", bg=BG_DARK,
                                   fg=TEXT_DIM, font=FONT_SMALL)
        self.lbl_estado.place(relx=0.75, rely=0.085)

        # ── Fila superior: métricas + matriz ─────────────────────────────────
        metrics_card = card(self)
        metrics_card.place(relx=0.01, rely=0.11, relwidth=0.34, relheight=0.46)
        section_label(metrics_card, "Métricas Globales")
        separator(metrics_card)
        self._build_metrics(metrics_card)

        cm_card = card(self)
        cm_card.place(relx=0.37, rely=0.11, relwidth=0.62, relheight=0.46)
        section_label(cm_card, "Matriz de Confusión")
        separator(cm_card)
        self._build_cm_area(cm_card)

        # ── Fila inferior: gráficas ───────────────────────────────────────────
        chart_card = card(self)
        chart_card.place(relx=0.01, rely=0.59, relwidth=0.98, relheight=0.40)
        section_label(chart_card, "Gráficas de Evaluación")
        separator(chart_card)
        self._build_charts_area(chart_card)

    def _build_metrics(self, parent):
        metrics_def = [
            ("Exactitud (Accuracy)",  ACCENT),
            ("Precisión (Precision)", ACCENT2),
            ("Sensibilidad (Recall)", SUCCESS),
            ("F1-Score",              WARNING),
            ("EG prueba (MAE)",       TEXT_DIM),
            ("Estado entren.",        TEXT_DIM),
        ]
        self.metric_labels: dict = {}
        for name, color in metrics_def:
            row = tk.Frame(parent, bg=BG_CARD)
            row.pack(fill="x", padx=12, pady=5)
            tk.Label(row, text=name, bg=BG_CARD, fg=TEXT_DIM,
                     font=FONT_SMALL, width=24, anchor="w").pack(side="left")
            lbl = tk.Label(row, text="—", bg=BG_CARD, fg=color,
                           font=("Segoe UI", 11, "bold"))
            lbl.pack(side="left")
            self.metric_labels[name] = lbl

    def _build_cm_area(self, parent):
        self._cm_parent = parent
        self._cm_holder = tk.Frame(parent, bg=BG_CARD)
        self._cm_holder.pack(fill="both", expand=True, padx=12, pady=4)
        tk.Label(self._cm_holder,
                 text="Execute el pipeline para ver la matriz de confusión.",
                 bg=BG_CARD, fg=TEXT_DIM, font=FONT_SMALL).pack(expand=True)

    def _build_charts_area(self, parent):
        self._chart_parent = parent
        self._chart_holder = tk.Frame(parent, bg=BG_CARD)
        self._chart_holder.pack(fill="both", expand=True, padx=12, pady=4)
        tk.Label(self._chart_holder,
                 text="Execute el pipeline para ver las gráficas de evaluación.",
                 bg=BG_CARD, fg=TEXT_DIM, font=FONT_SMALL).pack(expand=True)

    # ── Lógica ────────────────────────────────────────────────────────────────

    def on_show(self):
        if self.app_state.get("resultados") is not None:
            self._calcular()

    def _calcular(self):
        res      = self.app_state.get("resultados")
        hist_eg  = self.app_state.get("historial_EG",      [])
        hist_cen = self.app_state.get("historial_centros",  [])
        cfg      = self.app_state.get("config")
        convergio = self.app_state.get("convergio", False)

        if res is None:
            self.lbl_estado.configure(
                text="⚠  Sin resultados — ejecute el pipeline en Configuración.", fg=WARNING)
            self.status.set("Sin resultados. Configure y cargue un dataset primero.", "warn")
            return

        # ── Métricas ──────────────────────────────────────────────────────────
        self.metric_labels["Exactitud (Accuracy)"].configure(
            text=f"{res['exactitud']*100:.2f}%", fg=ACCENT)
        self.metric_labels["Precisión (Precision)"].configure(
            text=f"{res['precision']:.4f}", fg=ACCENT2)
        self.metric_labels["Sensibilidad (Recall)"].configure(
            text=f"{res['sensibilidad']:.4f}", fg=SUCCESS)
        self.metric_labels["F1-Score"].configure(
            text=f"{res['f1']:.4f}", fg=WARNING)
        self.metric_labels["EG prueba (MAE)"].configure(
            text=f"{res['EG_test']:.6f}", fg=TEXT_MAIN)
        estado_txt = "CONVERGE" if convergio else "NO CONVERGE"
        estado_color = SUCCESS if convergio else DANGER
        self.metric_labels["Estado entren."].configure(
            text=estado_txt, fg=estado_color)

        # ── Matriz de confusión ───────────────────────────────────────────────
        cm = res["confusion_matrix"]
        self._dibujar_cm(cm)

        # ── Gráficas ──────────────────────────────────────────────────────────
        error_optimo = cfg.error_optimo if cfg else 0.0
        self._dibujar_graficas(res, hist_eg, hist_cen, error_optimo)

        self.lbl_estado.configure(
            text=f"Exactitud: {res['exactitud']*100:.2f}%  |  "
                 f"F1: {res['f1']:.4f}", fg=SUCCESS)
        self.status.set(
            f"Evaluación — Exactitud: {res['exactitud']*100:.2f}%  F1: {res['f1']:.4f}", "ok")

    def _dibujar_cm(self, cm: np.ndarray):
        # Destruir canvas anterior
        for w in self._cm_holder.winfo_children():
            w.destroy()

        n_clases = cm.shape[0]
        fig = Figure(figsize=(max(4, n_clases * 1.4), max(3, n_clases * 1.1)),
                     facecolor=BG_CARD)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(BG_DARK)

        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.85)

        labels = [f"C{i}" for i in range(n_clases)]
        ax.set_xticks(range(n_clases))
        ax.set_yticks(range(n_clases))
        ax.set_xticklabels(labels, color=TEXT_DIM, fontsize=8)
        ax.set_yticklabels(labels, color=TEXT_DIM, fontsize=8)
        ax.set_xlabel("Predicho", color=TEXT_DIM, fontsize=9)
        ax.set_ylabel("Real",     color=TEXT_DIM, fontsize=9)
        ax.set_title("Matriz de confusión", color=TEXT_MAIN, fontsize=9)

        umbral = cm.max() / 2.0
        for i in range(n_clases):
            for j in range(n_clases):
                color = "white" if cm[i, j] > umbral else "black"
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")

        fig.tight_layout(pad=1.2)
        canvas = FigureCanvasTkAgg(fig, master=self._cm_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas_cm = canvas

    def _dibujar_graficas(self, res: dict, hist_eg: list,
                          hist_cen: list, error_optimo: float):
        # Destruir canvas anterior
        for w in self._chart_holder.winfo_children():
            w.destroy()

        Yd_clase = res["Yd_clase"]
        Yr_clase = res["Yr_clase"]
        EL       = res["EL"].ravel()
        n_test   = len(Yd_clase)
        x_idx    = np.arange(1, n_test + 1)

        fig = Figure(figsize=(13, 2.8), facecolor=BG_CARD)

        # Gráfica 1 — YD vs YR
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_facecolor(BG_DARK)
        ax1.plot(x_idx, Yd_clase, "o-", color=ACCENT,  label="YD (real)",
                 linewidth=1.5, markersize=3)
        ax1.plot(x_idx, Yr_clase, "s--", color=ACCENT2, label="YR (pred.)",
                 linewidth=1.5, markersize=3)
        ax1.set_title("YD vs YR", color=TEXT_MAIN, fontsize=9)
        ax1.set_xlabel("Patrón (prueba)", color=TEXT_DIM, fontsize=8)
        ax1.set_ylabel("Clase", color=TEXT_DIM, fontsize=8)
        ax1.legend(facecolor=BG_PANEL, labelcolor=TEXT_MAIN, fontsize=7)
        ax1.tick_params(colors=TEXT_DIM, labelsize=7)
        for sp in ax1.spines.values(): sp.set_edgecolor(BORDER)

        # Gráfica 2 — EG por iteración
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_facecolor(BG_DARK)
        if hist_eg and hist_cen:
            ax2.plot(hist_cen, hist_eg, "o-", color=SUCCESS,
                     linewidth=2, markersize=5, label="EG")
            ax2.axhline(y=error_optimo, color=DANGER, linestyle="--",
                        linewidth=1.5, label=f"Óptimo={error_optimo}")
            ax2.legend(facecolor=BG_PANEL, labelcolor=TEXT_MAIN, fontsize=7)
        ax2.set_title("EG por iteración", color=TEXT_MAIN, fontsize=9)
        ax2.set_xlabel("N° centros", color=TEXT_DIM, fontsize=8)
        ax2.set_ylabel("EG", color=TEXT_DIM, fontsize=8)
        ax2.tick_params(colors=TEXT_DIM, labelsize=7)
        for sp in ax2.spines.values(): sp.set_edgecolor(BORDER)

        # Gráfica 3 — |EL| por patrón
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_facecolor(BG_DARK)
        abs_el  = np.abs(EL)
        colores = [DANGER if v > error_optimo else ACCENT for v in abs_el]
        ax3.bar(x_idx, abs_el, color=colores, alpha=0.85,
                edgecolor=BG_DARK, linewidth=0.3)
        if error_optimo > 0:
            ax3.axhline(y=error_optimo, color=DANGER, linestyle="--",
                        linewidth=1.5, label=f"Óptimo={error_optimo}")
            ax3.legend(facecolor=BG_PANEL, labelcolor=TEXT_MAIN, fontsize=7)
        ax3.set_title("|EL| por patrón (prueba)", color=TEXT_MAIN, fontsize=9)
        ax3.set_xlabel("Patrón", color=TEXT_DIM, fontsize=8)
        ax3.set_ylabel("|EL|", color=TEXT_DIM, fontsize=8)
        ax3.tick_params(colors=TEXT_DIM, labelsize=7)
        for sp in ax3.spines.values(): sp.set_edgecolor(BORDER)

        fig.tight_layout(pad=1.4)
        canvas = FigureCanvasTkAgg(fig, master=self._chart_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas_charts = canvas
