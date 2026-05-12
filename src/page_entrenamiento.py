# -*- coding: utf-8 -*-
"""Página 4 – Parámetros y Entrenamiento del Modelo RBF."""

import threading
import tkinter as tk
from tkinter import filedialog

from core.model_io import save_model
from core.preprocessing import split_dataset
from panels.hiperparametros import PanelHiperparametros
from panels.resultados import PanelResultados
from core.evaluator import evaluar_modelo
from styles import (
    BG_DARK, TEXT_MAIN, TEXT_DIM, SUCCESS, WARNING, ACCENT2,
    FONT_TITLE, FONT_SMALL, ACCENT,
)
from widgets import card


class PageEntrenamiento(tk.Frame):
    def __init__(self, parent, status_bar, app_state):
        super().__init__(parent, bg=BG_DARK)
        self.status = status_bar
        self.app_state = app_state
        self._stop_flag = threading.Event()
        self._train_thread = None

        # Datos internos capturados tras el entrenamiento
        self._modelo = None
        self._hist_EG = []
        self._hist_cen = []
        self._centros = None
        self._pesos = None
        self._D_sample = None
        self._FA_sample = None
        self._A_sample = None
        self._X_train = None
        self._Yd_train = None
        self._col_names = []

        self._build()

    def _build(self):
        tk.Label(self, text="Parámetros y Entrenamiento del Modelo RBF",bg=BG_DARK, fg=TEXT_MAIN, font=FONT_TITLE).pack(anchor="w", padx=24, pady=(20, 2))
        tk.Label(self,text="Configure la arquitectura, entrene la red y explore las matrices internas.",bg=BG_DARK, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor="w", padx=24)

        # Panel izquierdo ── parámetros
        self.left_panel = PanelHiperparametros(
            self, self.app_state,
            on_entrenar=self._entrenar,
            on_detener=self._detener
        )
        self.left_panel.place(relx=0.01, rely=0.12, relwidth=0.30, relheight=0.86)

        # Panel derecho
        self.right_panel = PanelResultados(self, self.app_state)
        self.right_panel.place(relx=0.32, rely=0.12, relwidth=0.67, relheight=0.86)

    # ── Acciones ──────────────────────────────────────────────────────────────

    def on_show(self):
        self.left_panel.cargar_desde_config()

    def _detener(self):
        self._stop_flag.set()
        self.status.set("Deteniendo entrenamiento…", "warn")

    def _entrenar(self):
        # Validar que haya datos
        self.left_panel.deshabilitar_guardado()
        X   = self.app_state.get("X")
        Yd  = self.app_state.get("Yd")
        cfg = self.app_state.get("config")
        if X is None or cfg is None:
            self.status.set("Sin datos — configure el dataset primero.", "warn")
            # Actualizar estado en panel izquierdo
            self.left_panel.conv_lbl.configure(text="⚠ Sin datos", fg=WARNING)
            return
        self.left_panel.deshabilitar_guardado()
        # Obtener valores actuales del panel de hiperparámetros
        valores = self.left_panel.obtener_valores()

        # Aplicar parámetros de la UI a config
        try:
            cfg.n_centros        = int(valores["n_centros"])
            cfg.error_optimo     = float(valores["error_optimo"])
            cfg.max_iteraciones  = int(valores["max_iteraciones"])
        except ValueError:
            self.status.set("Parámetros inválidos.", "error")
            return

        cfg.tipo_particion = valores["particion"]
        cfg.random_state   = 42 if valores["semilla"] == "Fija (42)" else None

        # Re-partir el dataset con los parámetros actuales
        splits = split_dataset(X, Yd, cfg.tipo_particion, cfg.random_state)
        self.app_state["splits"] = splits

        self._col_names = list(cfg.input_columns)

        # Limpiar panel derecho y resetear estado del izquierdo
        self.right_panel.limpiar_todo()
        self.left_panel.resetear_estado()

        # Reiniciar estado
        self._stop_flag.clear()
        self._hist_EG  = []
        self._hist_cen = []

        # Hilo de entrenamiento
        self._train_thread = threading.Thread(
            target=self._run_training,
            args=(splits, cfg),
            daemon=True,
        )
        self._train_thread.start()

    def _run_training(self, splits, cfg):
        from src.core.trainer import entrenar_rbf
        info = self.app_state.get("info", {})
        min_X = info.get("min_X", 0.0)
        max_X = info.get("max_X", 1.0)

        def on_iter(iter_n, total, eg):
            if self._stop_flag.is_set():
                return
            pct = (iter_n / max(total, 1)) * 100
            # Pasar el error óptimo para la curva
            self.after(0, self._update_progress, iter_n, total, eg, pct, cfg.error_optimo)

        try:
            modelo, hist_EG, hist_cen, convergio = entrenar_rbf(
                splits["X_train"], splits["Yd_train"],
                cfg, min_X, max_X,
                verbose=False,
                on_iter_progress=on_iter,
            )
        except Exception as exc:
            self.after(0, self.status.set, f"Error: {exc}", "error")
            return

        # Capturar matrices internas
        self._modelo   = modelo
        self._hist_EG  = hist_EG
        self._hist_cen = hist_cen
        self._centros  = modelo.centros
        self._pesos    = modelo.pesos
        self._X_train  = splits["X_train"]
        self._Yd_train = splits["Yd_train"]

        # Reconstruir matrices con el modelo entrenado
        A, D, FA = modelo._construir_matriz_activacion(
            splits["X_train"], modelo.centros)
        self._A_sample  = A
        self._D_sample  = D
        self._FA_sample = FA

        # Guardar en app_state
        self.app_state["modelo"]            = modelo
        self.app_state["historial_EG"]      = hist_EG
        self.app_state["historial_centros"] = hist_cen
        self.app_state["convergio"]         = convergio
        resultados = evaluar_modelo(
            self._modelo,
            splits["X_test"],
            splits["Yd_test"],
            verbose=False,
            diagnostico=False
        )
        self.app_state["resultados"] = resultados
        self.after(0, self._on_training_done, hist_EG, hist_cen, convergio, cfg)

    def _update_progress(self, iter_n, total, eg, pct, error_optimo):
        self.left_panel.actualizar_progreso(iter_n, total, eg, pct)
        self.right_panel.actualizar_curva(self._hist_EG, error_optimo)

    def _guardar_modelo(self):
        modelo = self.app_state.get("modelo")
        cfg = self.app_state.get("config")
        info = self.app_state.get("info", {})
        if modelo is None:
            return

        ruta = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("Modelo RBF", "*.json"), ("Todos", "*.*")],
            title="Guardar modelo entrenado")
        if not ruta:
            return

        config_dict = cfg.to_dict() if hasattr(cfg, 'to_dict') else cfg
        save_model(modelo, config_dict, info, ruta)
        self.status.set(f"Modelo guardado en {ruta}", "ok")

    def _on_training_done(self, hist_EG, hist_cen, convergio, cfg):
        self.left_panel.entrenamiento_terminado(hist_EG, convergio)

        # Poblar resultados en panel derecho
        self.right_panel.actualizar_curva(hist_EG, cfg.error_optimo)
        self.right_panel.mostrar_centros(self._centros, self._col_names)
        self.right_panel.mostrar_distancias(self._D_sample)
        self.right_panel.mostrar_fa(self._FA_sample)
        self.right_panel.mostrar_A(self._A_sample)
        self.right_panel.mostrar_pesos(self._pesos)
        self.left_panel.habilitar_guardado(self._guardar_modelo)

        # Particiones
        splits = self.app_state["splits"]
        self.right_panel.mostrar_particion(
            "train", splits["X_train"], splits["Yd_train"],
            "Conjunto de Entrenamiento", SUCCESS)
        self.right_panel.mostrar_particion(
            "val", splits["X_val"], splits["Yd_val"],
            "Conjunto de Validación", WARNING)
        self.right_panel.mostrar_particion(
            "test", splits["X_test"], splits["Yd_test"],
            "Conjunto de Prueba", ACCENT2)

        self.status.set(
            f"Entrenamiento finalizado ({'convergido' if convergio else 'no convergido'}).",
            "ok" if convergio else "warn")
        
        self.left_panel.habilitar_guardado(self._guardar_modelo)
