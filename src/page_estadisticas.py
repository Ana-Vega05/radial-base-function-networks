import tkinter as tk

import numpy as np
from styles import BG_DARK, TEXT_MAIN, TEXT_DIM, SUCCESS, WARNING, ACCENT, FONT_TITLE, FONT_SMALL
from widgets import card, section_label, separator, primary_btn
from panels.resumen_estadisticas import PanelResumenEstadisticas
from panels.tabla_estadisticas import PanelTablaEstadisticas
from panels.histogramas import PanelHistogramas
from panels.distribucion_clases import PanelDistribucionClases

class PageEstadisticas(tk.Frame):
    def __init__(self, parent, status_bar, app_state: dict):
        super().__init__(parent, bg=BG_DARK)
        self.status = status_bar
        self.app_state = app_state
        self._build()

    def _build(self):
        tk.Label(self, text="Estadística Descriptiva", bg=BG_DARK,fg=TEXT_MAIN, font=FONT_TITLE).pack(anchor="w", padx=24, pady=(20, 2))
        tk.Label(self, text="Resumen estadístico, distribución de variables e información del dataset cargado.",bg=BG_DARK, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor="w", padx=24, pady=(0, 8))

        primary_btn(self, "↺  Actualizar", self._calcular).place(relx=0.80, rely=0.02)
        self.lbl_estado = tk.Label(self, text="", bg=BG_DARK, fg=TEXT_DIM, font=FONT_SMALL)
        self.lbl_estado.place(relx=0.80, rely=0.075)

        # Tarjetas de resumen
        summary_card = card(self)
        summary_card.place(relx=0.01, rely=0.11, relwidth=0.98, relheight=0.10)
        self.panel_resumen = PanelResumenEstadisticas(summary_card)
        self.panel_resumen.pack(fill="both", expand=True)

        # Tabla estadística
        table_card = card(self)
        table_card.place(relx=0.01, rely=0.22, relwidth=0.98, relheight=0.33)
        section_label(table_card, "Tabla de Estadísticas por Variable")
        separator(table_card)
        self.panel_tabla = PanelTablaEstadisticas(table_card)
        self.panel_tabla.pack(fill="both", expand=True)

        # Histogramas
        hist_card = card(self)
        hist_card.place(relx=0.01, rely=0.56, relwidth=0.68, relheight=0.42)
        section_label(hist_card, "Distribución de Variables de Entrada")
        separator(hist_card)
        self.panel_hist = PanelHistogramas(hist_card)
        self.panel_hist.pack(fill="both", expand=True)

        # Distribución de clases
        clase_card = card(self)
        clase_card.place(relx=0.70, rely=0.56, relwidth=0.29, relheight=0.42)
        section_label(clase_card, "Distribución de Clases")
        separator(clase_card)
        self.panel_clases = PanelDistribucionClases(clase_card)
        self.panel_clases.pack(fill="both", expand=True)

    def on_show(self):
        if self.app_state.get("X") is not None:
            self._calcular()

    def _calcular(self):
        X = self.app_state.get("X")
        Yd = self.app_state.get("Yd")
        config = self.app_state.get("config")

        if X is None or config is None:
            self.lbl_estado.configure(text="  Sin dataset — cargue datos en Configuración.", fg=WARNING)
            self.status.set("Sin datos — configure el dataset primero.", "warn")
            return

        input_cols = list(config.input_columns)
        target_col = config.target_column

        # Clases
        Yd_clase = Yd.ravel().astype(int) if Yd.shape[1] == 1 else np.argmax(Yd, axis=1)
        n_clases = len(np.unique(Yd_clase))

        self.panel_resumen.actualizar(X.shape[0], X.shape[1], Yd.shape[1], n_clases)
        self.panel_tabla.actualizar(X, Yd, input_cols, target_col)
        self.panel_hist.actualizar(X, input_cols)
        self.panel_clases.actualizar(Yd_clase, n_clases)

        self.lbl_estado.configure(text=f"  {X.shape[1]} variables · {X.shape[0]} patrones", fg=SUCCESS)
        self.status.set("Estadísticas calculadas correctamente.", "ok")