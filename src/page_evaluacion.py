import tkinter as tk
from styles import BG_DARK, TEXT_MAIN, TEXT_DIM, SUCCESS, WARNING, ACCENT,FONT_TITLE, FONT_SMALL
from widgets import card, section_label, separator, primary_btn
from panels.metricas_tarjetas import PanelMetricasTarjetas
from panels.reporte_clase import PanelReporteClase
from panels.matriz_confusion import PanelMatrizConfusion
from panels.graficas_evaluacion import PanelGraficasEvaluacion

class PageEvaluacion(tk.Frame):
    def __init__(self, parent, status_bar, app_state: dict):
        super().__init__(parent, bg=BG_DARK)
        self.status = status_bar
        self.app_state = app_state
        self._build()

    def _build(self):
        tk.Label(self, text="Evaluación del Modelo", bg=BG_DARK,
                 fg=TEXT_MAIN, font=FONT_TITLE).pack(anchor="w", padx=24, pady=(20, 2))
        tk.Label(self, text="Métricas de clasificación, matriz de confusión y gráficas de rendimiento.",
                 bg=BG_DARK, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor="w", padx=24, pady=(0, 6))

        primary_btn(self, "↺  Actualizar", self._calcular).place(relx=0.80, rely=0.02)
        self.lbl_estado = tk.Label(self, text="", bg=BG_DARK, fg=TEXT_DIM, font=FONT_SMALL)
        self.lbl_estado.place(relx=0.80, rely=0.075)

        # Tarjetas de métricas
        metrics_card = card(self)
        metrics_card.place(relx=0.01, rely=0.11, relwidth=0.98, relheight=0.13)
        self.panel_metricas = PanelMetricasTarjetas(metrics_card)
        self.panel_metricas.pack(fill="both", expand=True)

        # Reporte por clase
        report_card = card(self)
        report_card.place(relx=0.01, rely=0.25, relwidth=0.42, relheight=0.36)
        section_label(report_card, "Reporte por Clase")
        separator(report_card)
        self.panel_reporte = PanelReporteClase(report_card)
        self.panel_reporte.pack(fill="both", expand=True)

        # Matriz de confusión
        cm_card = card(self)
        cm_card.place(relx=0.44, rely=0.25, relwidth=0.55, relheight=0.36)
        section_label(cm_card, "Matriz de Confusión")
        separator(cm_card)
        self.panel_cm = PanelMatrizConfusion(cm_card)
        self.panel_cm.pack(fill="both", expand=True)

        # Gráficas
        chart_card = card(self)
        chart_card.place(relx=0.01, rely=0.62, relwidth=0.98, relheight=0.37)
        section_label(chart_card, "Gráficas de Evaluación")
        separator(chart_card)
        self.panel_graficas = PanelGraficasEvaluacion(chart_card)
        self.panel_graficas.pack(fill="both", expand=True)

    def on_show(self):
        if self.app_state.get("resultados") is not None:
            self._calcular()

    def _calcular(self):
        res = self.app_state.get("resultados")
        hist_eg = self.app_state.get("historial_EG", [])
        hist_cen = self.app_state.get("historial_centros", [])
        cfg = self.app_state.get("config")
        convergio = self.app_state.get("convergio", False)

        if res is None:
            self.lbl_estado.configure(text="⚠  Sin resultados — entrene el modelo en Entrenamiento.", fg=WARNING)
            self.status.set("Sin resultados. Entrene el modelo primero.", "warn")
            return

        self.panel_metricas.actualizar(res, convergio)
        self.panel_reporte.actualizar(res)
        self.panel_cm.actualizar(res["confusion_matrix"])

        error_optimo = cfg.error_optimo if cfg else 0.0
        self.panel_graficas.actualizar(res, hist_eg, hist_cen, error_optimo)

        self.lbl_estado.configure(
            text=f"✓  Exactitud: {res['exactitud']*100:.2f}%  |  F1: {res['f1']:.4f}",
            fg=SUCCESS)
        self.status.set(f"Evaluación — Exactitud: {res['exactitud']*100:.2f}%  F1: {res['f1']:.4f}", "ok")