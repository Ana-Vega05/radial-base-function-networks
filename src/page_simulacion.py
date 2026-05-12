# -*- coding: utf-8 -*-
"""Página – Simulación del Modelo RBF (refactorizada)."""

import tkinter as tk
import numpy as np

from styles import (
    BG_DARK, TEXT_MAIN, TEXT_DIM, SUCCESS, WARNING, FONT_TITLE, FONT_SMALL
)
from widgets import card
from panels.simulacion_left import PanelSimulacionLeft
from panels.simulacion_right import PanelSimulacionRight


class PageSimulacion(tk.Frame):
    def __init__(self, parent, status_bar, app_state: dict):
        super().__init__(parent, bg=BG_DARK)
        self.status = status_bar
        self.app_state = app_state
        self._build()

    def _build(self):
        tk.Label(self, text="Simulación del Modelo",bg=BG_DARK, fg=TEXT_MAIN, font=FONT_TITLE).pack(anchor="w", padx=24, pady=(20, 2))
        tk.Label(self,text="Ingrese valores de entrada para obtener la predicción del modelo RBF entrenado.",bg=BG_DARK, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor="w", padx=24)

        # Panel izquierdo
        self.left = PanelSimulacionLeft(
            self, self.app_state,
            on_simular=self._simular,
            on_limpiar=self._limpiar)
        self.left.place(relx=0.01, rely=0.12, relwidth=0.36, relheight=0.86)

        # Panel derecho
        self.right = PanelSimulacionRight(self)
        self.right.place(relx=0.38, rely=0.12, relwidth=0.61, relheight=0.86)

        # Configurar evento de cambio de modo
        self.left.mode_var.trace_add("write", self._on_mode_change)
        # Configurar comando del botón "Cargar patrón"
        self.left.set_load_patron_command(self._cargar_patron)

    def on_show(self):
        self.left.build_entries()  # actualiza las entradas según la configuración

    # Manejo de modo y patrón

    def _on_mode_change(self, *_):
        modo = self.left.mode_var.get()
        if modo == "Entrada manual":
            self.left.show_pattern_controls(False)
        else:
            self.left.show_pattern_controls(True)
            self._actualizar_limite_spin()
            # Cargar automáticamente el primer patrón
            self._cargar_patron()

    def _actualizar_limite_spin(self):
        modo = self.left.mode_var.get()
        splits = self.app_state.get("splits", {})
        key = "X_test" if modo == "Conjunto de prueba" else "X_val"
        Xp = splits.get(key)
        limite = Xp.shape[0] if Xp is not None else 1
        self.left.set_spin_max(limite)

    def _cargar_patron(self):
        modo = self.left.mode_var.get()
        splits = self.app_state.get("splits", {})
        key_X = "X_test" if modo == "Conjunto de prueba" else "X_val"
        key_Y = "Yd_test" if modo == "Conjunto de prueba" else "Yd_val"
        Xp = splits.get(key_X)
        Yp = splits.get(key_Y)

        if Xp is None:
            self.status.set("Sin particiones — entrene el modelo primero.", "warn")
            return

        idx = self.left.get_spin_value() - 1
        idx = max(0, min(idx, Xp.shape[0] - 1))
        patron = Xp[idx]
        cfg = self.app_state.get("config")
        cols = list(cfg.input_columns) if cfg else list(self.left.sim_entries.keys())

        self.left.set_entries_from_patron(patron, cols)

        # Clase real
        if Yp is not None and Yp.shape[0] > idx:
            if Yp.shape[1] == 1:
                clase_real = int(Yp[idx, 0])
            else:
                clase_real = int(np.argmax(Yp[idx]))
            self.right.set_real_class(clase_real)
        else:
            self.right.clear_real_class()

    # Simulación

    def _simular(self):
        modelo = self.app_state.get("modelo")
        cfg = self.app_state.get("config")

        if modelo is None:
            self.status.set("Sin modelo — entrene primero en la pestaña Entrenamiento.", "warn")
            self.right.clear()
            return

        try:
            entrada = np.array(self.left.get_entry_values(), dtype=np.float64).reshape(1, -1)
        except (ValueError, KeyError):
            self.status.set("Valores de entrada inválidos.", "error")
            return

        Yr = modelo.predict(entrada)
        n_salidas = Yr.shape[1]

        if n_salidas == 1:
            score = float(Yr[0, 0])
            predicted = 1 if score >= 0.5 else 0
            scores = np.array([1.0 - score, score])
            clases = ["Clase 0", "Clase 1"]
            conf = max(scores)
        else:
            scores = Yr[0]
            predicted = int(np.argmax(scores))
            sc_min, sc_max = scores.min(), scores.max()
            if sc_max > sc_min:
                probs = (scores - sc_min) / (sc_max - sc_min)
            else:
                probs = np.ones_like(scores) / len(scores)
            scores = probs
            clases = [f"Clase {k}" for k in range(n_salidas)]
            conf = float(scores[predicted])

        self.right.set_prediction(predicted, conf)
        self.right.update_bar_chart(clases, scores, predicted)
        self.right.show_yr_raw(Yr[0], clases)

        self.status.set(f"Predicción: Clase {predicted}  (score = {conf:.4f})", "ok")

    # Limpiar

    def _limpiar(self):
        self.left.clear_entries()
        self.right.clear()
        self.status.set("Campos limpios.")