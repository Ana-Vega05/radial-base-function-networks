# -*- coding: utf-8 -*-
"""Panel derecho – Resultados del entrenamiento (gráfica y matrices)."""

import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from styles import (
    BG_CARD, BG_DARK, BG_PANEL, ACCENT, ACCENT2, FONT_HEADER, TEXT_MAIN, TEXT_DIM,
    SUCCESS, DANGER, BORDER, FONT_BODY, FONT_SMALL, WARNING,
)
from src.widgets.tables import matriz_a_treeview, vector_a_treeview
from widgets import section_label, separator


class PanelResultados(tk.Frame):
    """Contiene el Notebook con las pestañas de resultados y sus gráficos/tablas."""

    def __init__(self, parent, app_state: dict, **kwargs):
        super().__init__(parent, bg=BG_CARD, **kwargs)
        self.app_state = app_state
        self._col_names = []

        # Atributos para la pestaña de curva EG
        self.fig_loss = None
        self.ax_loss = None
        self.canvas_loss = None

        self._build()

    #Construcción del Notebook

    def _build(self):
        section_label(self, "Resultados del Entrenamiento")
        separator(self)

        self.nb = ttk.Notebook(self, style="TNotebook")
        self.nb.pack(fill="both", expand=True, padx=8, pady=6)

        # Crear las pestañas
        self._tab_curva   = tk.Frame(self.nb, bg=BG_CARD)
        self._tab_centros = tk.Frame(self.nb, bg=BG_CARD)
        self._tab_D       = tk.Frame(self.nb, bg=BG_CARD)
        self._tab_FA      = tk.Frame(self.nb, bg=BG_CARD)
        self._tab_A       = tk.Frame(self.nb, bg=BG_CARD)
        self._tab_W       = tk.Frame(self.nb, bg=BG_CARD)
        self._tab_train   = tk.Frame(self.nb, bg=BG_CARD)
        self._tab_val     = tk.Frame(self.nb, bg=BG_CARD)
        self._tab_test    = tk.Frame(self.nb, bg=BG_CARD)

        for tab, title in [
            (self._tab_curva,   "📈 Curva EG"),
            (self._tab_centros, "🎯 Centros R"),
            (self._tab_D,       "📐 Distancias D"),
            (self._tab_FA,      "⚡ FA(D)"),
            (self._tab_A,       "🔢 Matriz A"),
            (self._tab_W,       "⚖️ Pesos W"),
            (self._tab_train,   "🟢 Train"),
            (self._tab_val,     "🟡 Val"),
            (self._tab_test,    "🔵 Test"),
        ]:
            self.nb.add(tab, text=title)
            self._placeholder(tab)

        # Inicializar la gráfica de EG
        self._build_curva_plot()

    # ── Métodos de ayuda para las pestañas

    def _placeholder(self, parent, msg="Entrene el modelo para ver los datos."):
        tk.Label(parent, text=msg, bg=BG_CARD, fg=TEXT_DIM,font=FONT_SMALL).pack(expand=True)

    def _limpiar_tab(self, tab):
        for w in tab.winfo_children():
            w.destroy()

    def _header(self, tab, texto, detalle=""):
        tk.Label(tab, text=texto, bg=BG_CARD, fg=ACCENT,font=FONT_HEADER).pack(anchor="w", padx=10, pady=(8, 0))
        if detalle:
            tk.Label(tab, text=detalle, bg=BG_CARD, fg=TEXT_DIM,font=FONT_SMALL).pack(anchor="w", padx=10, pady=(0, 4))
        tk.Frame(tab, bg=BORDER, height=1).pack(fill="x", padx=10, pady=2)

    def _poblar_matriz(self, tab, matrix, titulo, detalle,col_prefix="C", row_prefix="P", max_rows=200):
        self._limpiar_tab(tab)
        n_f, n_c = matrix.shape
        self._header(tab, f"{titulo}  [{n_f} × {n_c}]", detalle)
        matriz_a_treeview(tab, matrix, col_prefix=col_prefix,row_prefix=row_prefix, max_rows=max_rows)

    def limpiar_todo(self):
        """Limpia todas las pestañas excepto la de la curva (que se reconstruye)."""
        for tab in [self._tab_centros, self._tab_D, self._tab_FA,
                    self._tab_A, self._tab_W, self._tab_train,
                    self._tab_val, self._tab_test]:
            self._limpiar_tab(tab)
            self._placeholder(tab)
        self._build_curva_plot()   # reconstruye el lienzo

    # Pestaña Curva EG 

    def _build_curva_plot(self):
        """Crea la figura y el canvas para la curva de error."""
        for w in self._tab_curva.winfo_children():
            w.destroy()
        self.fig_loss = Figure(figsize=(7, 3.8), facecolor=BG_CARD)
        self.ax_loss = self.fig_loss.add_subplot(111)
        self.ax_loss.set_facecolor(BG_DARK)
        self.ax_loss.set_xlabel("Iteración", color=TEXT_DIM, fontsize=9)
        self.ax_loss.set_ylabel("EG", color=TEXT_DIM, fontsize=9)
        self.ax_loss.set_title("Error General por Iteración",color=TEXT_MAIN, fontsize=10)
        self.ax_loss.tick_params(colors=TEXT_DIM)
        for spine in self.ax_loss.spines.values():
            spine.set_edgecolor(BORDER)
        self.fig_loss.tight_layout(pad=1.5)
        self.canvas_loss = FigureCanvasTkAgg(self.fig_loss, master=self._tab_curva)
        self.canvas_loss.draw()
        self.canvas_loss.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=6)

    def actualizar_curva(self, hist_EG, error_optimo):
        """Redibuja la curva con el historial de EG."""
        if not hist_EG or self.ax_loss is None:
            return
        self.ax_loss.clear()
        self.ax_loss.set_facecolor(BG_DARK)
        iters = list(range(1, len(hist_EG) + 1))
        self.ax_loss.plot(iters, hist_EG, "o-",color=ACCENT, linewidth=2, markersize=5, label="EG")
        self.ax_loss.axhline(y=error_optimo, color=DANGER, linestyle="--",linewidth=1.5, label=f"Óptimo = {error_optimo}")

        # Marcar punto de convergencia
        for i, eg in enumerate(hist_EG):
            if eg <= error_optimo:
                self.ax_loss.plot(i + 1, eg, "D", color=SUCCESS,markersize=9, zorder=5,label=f"Converge iter {i+1}")
                break

        self.ax_loss.set_xlabel("Iteración", color=TEXT_DIM, fontsize=9)
        self.ax_loss.set_ylabel("EG", color=TEXT_DIM, fontsize=9)
        self.ax_loss.set_title("Error General por Iteración",color=TEXT_MAIN, fontsize=10)
        self.ax_loss.legend(facecolor=BG_PANEL, labelcolor=TEXT_MAIN, fontsize=8)
        self.ax_loss.tick_params(colors=TEXT_DIM)
        for spine in self.ax_loss.spines.values():
            spine.set_edgecolor(BORDER)
        self.fig_loss.tight_layout(pad=1.5)
        self.canvas_loss.draw()

    #Metodos para poblar las demas pestañas 

    def mostrar_centros(self, centros, col_names):
        n_c, n_e = centros.shape
        self._poblar_matriz(self._tab_centros, centros,
                            "Centros Radiales R",
                            f"{n_c} centros  ·  {n_e} entradas por centro",
                            col_prefix="", row_prefix="R")

    def mostrar_distancias(self, D):
        n_p, n_c = D.shape
        self._poblar_matriz(self._tab_D, D,
                            "Matriz de Distancias D",
                            "D[i,j] = distancia euclidiana entre patrón i y centro j",
                            col_prefix="R", row_prefix="P")

    def mostrar_fa(self, FA):
        n_p, n_c = FA.shape
        self._poblar_matriz(self._tab_FA, FA,
                            "Activaciones FA(D)",
                            "FA(D) = D² · ln(D)  (función de base radial de placa delgada)",
                            col_prefix="R", row_prefix="P")

    def mostrar_A(self, A):
        n_p, n_cols = A.shape
        self._poblar_matriz(self._tab_A, A,
                            "Matriz de Interpolación A",
                            "A = [1 | FA(D)]  (primera columna = bias)",
                            col_prefix="FA", row_prefix="P")

    def mostrar_pesos(self, pesos):
        tab = self._tab_W
        self._limpiar_tab(tab)
        W = pesos
        n_w, n_s = W.shape
        n_c = n_w - 1
        self._header(tab, f"Pesos W  [{n_w} × {n_s}]","W = pinv(A) · Yd   (fila 0 = bias Wo, filas 1…k = Wi)")
        etiquetas_fila = ["Wo (bias)"] + [f"W{j+1}" for j in range(n_c)]
        etiquetas_col  = [f"Sal{k}" for k in range(n_s)] if n_s > 1 else ["Salida"]
        vector_a_treeview(tab, W, etiquetas_fila, etiquetas_col)

    def mostrar_particion(self, tab_key, X, Yd, titulo, color):
        """Muestra una partición concreta (train/val/test) en la pestaña indicada."""
        tab_map = {
            "train": self._tab_train,
            "val":   self._tab_val,
            "test":  self._tab_test,
        }
        tab = tab_map[tab_key]
        self._limpiar_tab(tab)
        n, _ = X.shape
        self._header(tab, f"{titulo}  ({n} patrones)", "")

        info_frame = tk.Frame(tab, bg=BG_CARD)
        info_frame.pack(fill="x", padx=10, pady=2)
        tk.Label(info_frame, text=f"Patrones: {n}",
                bg=BG_CARD, fg=color, font=FONT_BODY).pack(side="left", padx=6)
        if Yd.shape[1] == 1:
            clases, cnts = np.unique(Yd.ravel().astype(int), return_counts=True)
            dist = "  ".join(f"C{c}: {n_}" for c, n_ in zip(clases, cnts))
            tk.Label(info_frame, text=f"Distribución clases → {dist}",
                    bg=BG_CARD, fg=TEXT_DIM, font=FONT_SMALL).pack(side="left", padx=8)

        data = np.hstack([X, Yd])
        matriz_a_treeview(tab, data, col_prefix="", row_prefix="P")


    # Método opcional para configurar los nombres de columnas si vienen de fuera
    def set_col_names(self, names):
        self._col_names = names