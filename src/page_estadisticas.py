# -*- coding: utf-8 -*-
"""Vista de Estadísticas — tabla descriptiva (sin Q1/Q3) e histogramas reales."""

import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from styles import (
    BG_DARK, BG_CARD, BG_PANEL, ACCENT, TEXT_MAIN, TEXT_DIM,
    SUCCESS, WARNING, BORDER, FONT_TITLE, FONT_SMALL,
)
from widgets import card, section_label, separator, action_btn


class PageEstadisticas(tk.Frame):
    def __init__(self, parent, status_bar, app_state: dict):
        super().__init__(parent, bg=BG_DARK)
        self.status    = status_bar
        self.app_state = app_state
        self._canvas   = None
        self._axes     = []
        self._fig      = None
        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        tk.Label(self, text="Estadística Descriptiva", bg=BG_DARK,
                 fg=TEXT_MAIN, font=FONT_TITLE).pack(
                     anchor="w", padx=24, pady=(20, 4))
        tk.Label(self,
                 text="Resumen estadístico y distribución de variables del dataset cargado.",
                 bg=BG_DARK, fg=TEXT_DIM, font=FONT_SMALL).pack(
                     anchor="w", padx=24)

        # ── Tabla ─────────────────────────────────────────────────────────────
        top = card(self)
        top.place(relx=0.01, rely=0.12, relwidth=0.98, relheight=0.40)

        header_row = tk.Frame(top, bg=BG_CARD)
        header_row.pack(fill="x", padx=12, pady=(4, 0))
        section_label(top, "Tabla de Estadísticas")
        separator(top)

        # Columnas: sin Q1 ni Q3
        cols = ("Variable", "Media", "Desv. Std", "Mín", "Mediana", "Máx")
        self.stat_tree = ttk.Treeview(
            top, columns=cols, show="headings",
            style="Dark.Treeview", height=7)

        col_widths = {"Variable": 140, "Media": 100, "Desv. Std": 100,
                      "Mín": 90, "Mediana": 100, "Máx": 90}
        for c in cols:
            self.stat_tree.heading(c, text=c)
            self.stat_tree.column(c, width=col_widths.get(c, 90), anchor="center")
        self.stat_tree.pack(fill="both", expand=True, padx=12, pady=(4, 2))

        btn_row = tk.Frame(top, bg=BG_CARD)
        btn_row.pack(fill="x", padx=12, pady=(2, 6))
        action_btn(btn_row, "Calcular Estadísticas",
                   command=self._calcular, color=ACCENT).pack(side="left")
        self.lbl_estado = tk.Label(btn_row, text="", bg=BG_CARD,
                                   fg=TEXT_DIM, font=FONT_SMALL)
        self.lbl_estado.pack(side="left", padx=12)

        # ── Histogramas ───────────────────────────────────────────────────────
        bottom = card(self)
        bottom.place(relx=0.01, rely=0.54, relwidth=0.98, relheight=0.44)
        section_label(bottom, "Distribución de Variables")
        separator(bottom)
        self._build_plot_area(bottom)

    def _build_plot_area(self, parent):
        """Crea el área de histograma (se actualiza en _dibujar_histogramas)."""
        self._plot_parent = parent
        self._plot_holder = tk.Frame(parent, bg=BG_CARD)
        self._plot_holder.pack(fill="both", expand=True, padx=12, pady=4)
        self._placeholder = tk.Label(
            self._plot_holder,
            text="Cargue un dataset para ver la distribución de variables.",
            bg=BG_CARD, fg=TEXT_DIM, font=FONT_SMALL)
        self._placeholder.pack(expand=True)

    # ── Lógica ────────────────────────────────────────────────────────────────

    def on_show(self):
        """Llamado automáticamente al mostrar esta página."""
        if self.app_state.get("X") is not None:
            self._calcular()

    def _calcular(self):
        X      = self.app_state.get("X")
        Yd     = self.app_state.get("Yd")
        config = self.app_state.get("config")

        if X is None or config is None:
            self.lbl_estado.configure(
                text="⚠  No hay dataset cargado. Configure primero.", fg=WARNING)
            self.status.set("Sin datos — vaya a Configuración primero.", "warn")
            return

        # Combinar entradas + salida en un solo array
        cols     = list(config.input_columns) + [config.target_column]
        data_all = np.hstack([X, Yd])

        # Limpiar tabla
        for row in self.stat_tree.get_children():
            self.stat_tree.delete(row)

        for i, col in enumerate(cols):
            d = data_all[:, i]
            self.stat_tree.insert("", "end", values=(
                col,
                f"{np.mean(d):.4f}",
                f"{np.std(d):.4f}",
                f"{np.min(d):.4f}",
                f"{np.median(d):.4f}",
                f"{np.max(d):.4f}",
            ))

        n_vars = min(len(config.input_columns), 8)
        self._dibujar_histogramas(X[:, :n_vars], config.input_columns[:n_vars])

        self.lbl_estado.configure(
            text=f" {len(cols)} variables calculadas.", fg=SUCCESS)
        self.status.set("Estadísticas calculadas correctamente.", "ok")

    def _dibujar_histogramas(self, X: np.ndarray, nombres: list):
        """Dibuja o actualiza los histogramas de las columnas de entrada."""
        n = X.shape[1]
        ncols = min(n, 4)
        nrows = (n + ncols - 1) // ncols

        # Destruir canvas anterior si existe
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None

        # Ocultar placeholder
        self._placeholder.pack_forget()

        figw = max(9, ncols * 2.8)
        figh = max(2.5, nrows * 2.2)

        self._fig = Figure(figsize=(figw, figh), facecolor=BG_CARD)
        self._axes = []

        for i in range(n):
            ax = self._fig.add_subplot(nrows, ncols, i + 1)
            ax.hist(X[:, i], bins=25, color=ACCENT, edgecolor=BG_DARK, alpha=0.85)
            ax.set_title(nombres[i], color=TEXT_DIM, fontsize=7, pad=3)
            ax.set_facecolor(BG_DARK)
            ax.tick_params(colors=TEXT_DIM, labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
            self._axes.append(ax)

        self._fig.tight_layout(pad=1.2)
        self._canvas = FigureCanvasTkAgg(self._fig, master=self._plot_holder)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
