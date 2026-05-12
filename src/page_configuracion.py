# -*- coding: utf-8 -*-
"""Vista de Configuración — carga de dataset y parámetros RBF (sin entrenar)."""

import os
import threading
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog

from core.model_io import load_model
from styles import (
    BG_DARK, BG_CARD, BG_DARK, TEXT_MAIN, TEXT_DIM,
    SUCCESS, FONT_TITLE, FONT_SMALL, FONT_MONO,
)
from widgets import card, section_label, separator
from panels.configuracion import PanelConfiguracion

from core.config import Config
from core.data_loader import load_json_dataset
from core.preprocessing import split_dataset


class PageConfiguracion(tk.Frame):
    def __init__(self, parent, status_bar, app_state: dict):
        super().__init__(parent, bg=BG_DARK)
        self.status = status_bar
        self.app_state = app_state
        self._running = False
        self._build()

    def _build(self):
        tk.Label(self, text="Configuración de la Red RBF", bg=BG_DARK,fg=TEXT_MAIN, font=FONT_TITLE).pack(anchor="w", padx=24, pady=(20, 2))
        tk.Label(self,text="Seleccione el dataset y defina los parámetros de la red. ""El entrenamiento se realiza en la página 'Entrenamiento'.",bg=BG_DARK, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor="w", padx=24)

        # Panel izquierdo – dataset + hiperparámetros
        self.left = PanelConfiguracion(self, self.app_state,on_cargar_datos=self._cargar_datos)
        self.left.place(relx=0.01, rely=0.12, relwidth=0.45, relheight=0.86)

        # Panel derecho – resumen
        right = card(self)
        right.place(relx=0.48, rely=0.12, relwidth=0.51, relheight=0.86)

        section_label(right, "Resumen de configuración")
        separator(right)

        text_frame = tk.Frame(right, bg=BG_CARD)
        text_frame.pack(fill="both", expand=True, padx=12, pady=8)

        self.summary_text = tk.Text(
            text_frame, bg=BG_DARK, fg=TEXT_MAIN, font=FONT_MONO,
            relief="flat", state="disabled", wrap="word",
            insertbackground=TEXT_MAIN)
        scroll = tk.Scrollbar(text_frame, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.summary_text.pack(side="left", fill="both", expand=True)

        self._show_summary("Seleccione un dataset y configure los parámetros.\n\n""Haga clic en 'Elección aleatoria' para valores de ejemplo,\n""o ingrese sus propios valores y pulse 'Cargar datos'.")

        self.left.set_cargar_modelo_callback(self._cargar_modelo_guardado)


    def on_show(self):
        pass

    # Botón "Cargar datos" (solo carga, no entrena) 
    def _cargar_datos(self):
        if self._running:
            return
        path = self.left.get_current_path()
        if not path:
            messagebox.showerror("Error", "Seleccione un dataset primero.")
            return

        features = self.left.get_features() or [
            f"x{i+1}" for i in range(self.left.get_n_inputs())]
        nombre = os.path.splitext(os.path.basename(path))[0]

        # Valores por defecto para que no se requieran en esta página
        config_dict = {
            "raw_data_path": os.path.dirname(path),
            "processed_data_path": os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "processed"),
            "dataset_name": nombre,
            "input_columns": features,
            "target_column": "y",
            "n_centros": max(1, self.left.get_n_inputs()),   # por defecto = n_entradas
            "error_optimo": 0.01,
            "max_iteraciones": 50,
            "tipo_particion": "70-15-15",
            "random_state": None,
        }

        self._running = True
        self.left.set_buttons_state("disabled")
        self._show_summary("⏳  Cargando dataset y preparando particiones...")
        self.status.set("Cargando datos…", "warn")

        t = threading.Thread(target=self._run_load_thread,args=(config_dict,), daemon=True)
        t.start()

    def _run_load_thread(self, config_dict):
        try:
            config = Config.from_dict(config_dict)
            X, Yd, info = load_json_dataset(config)
            splits = split_dataset(X, Yd, config.tipo_particion, None)

            payload = {
                "config": config,
                "X": X,
                "Yd": Yd,
                "info": info,
                "splits": splits,
            }
            self.after(0, lambda: self._on_done(payload))
        except Exception as e:
            error_msg = str(e)              # Capturamos el mensaje aquí
            self.after(0, lambda: self._on_error(error_msg))

    def _on_done(self, payload: dict):
        self.app_state.update(payload)
        self._running = False
        self.left.set_buttons_state("normal")

        cfg = payload["config"]
        info = payload["info"]
        feat_str = ", ".join(cfg.input_columns)

        resumen = (
            "═══════════════════════════════════════\n"
            "  DATOS CARGADOS EXITOSAMENTE\n"
            "═══════════════════════════════════════\n\n"
            f"  Dataset         : {cfg.dataset_name}\n"
            f"  Entradas ({info['n_entradas']})   : {feat_str}\n"
            f"  Salidas         : {info['n_salidas']}\n"
            f"  Patrones totales: {info['n_patrones']}\n\n"
            "Los datos están listos. Vaya a la pestaña\n"
            "'Entrenamiento' para configurar el modelo\n"
            "y entrenar la red.\n"
        )
        self._show_summary(resumen)
        self.status.set("Dataset cargado correctamente.", "ok")

    def _on_error(self, msg: str):
        self._running = False
        self.left.set_buttons_state("normal")
        self._show_summary(f"❌  Error durante la carga:\n\n{msg}")
        self.status.set(f"Error: {msg[:80]}", "error")
        messagebox.showerror("Error en la carga", msg)

    def _show_summary(self, text: str):
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("end", text)
        self.summary_text.configure(state="disabled")

    def _cargar_modelo_guardado(self):
        ruta = filedialog.askopenfilename(
            filetypes=[("Modelo RBF", "*.json"), ("Todos", "*.*")],
            title="Cargar modelo entrenado")
        if not ruta:
            return

        try:
            modelo, config_dict, info = load_model(ruta)
        except Exception as e:
            messagebox.showerror("Error al cargar modelo", str(e))
            return

        # Reconstruir configuración
        config = Config.from_dict(config_dict)
        
        self.app_state["modelo"] = modelo
        self.app_state["config"] = config
        self.app_state["info"] = {**info, "n_entradas": config.n_entradas,
                                "n_salidas": config.n_salidas,
                                "n_patrones": 0, "n_clases": config.n_salidas}
        # No hay X, Yd, splits
        self.app_state["X"] = None
        self.app_state["Yd"] = None
        self.app_state["splits"] = None
        self.app_state["resultados"] = None

        self._show_summary(
            "═══════════════════════════════════════\n"
            "  MODELO CARGADO EXITOSAMENTE\n"
            "═══════════════════════════════════════\n\n"
            f"  Dataset original: {config.dataset_name}\n"
            f"  Entradas: {config.n_entradas}\n"
            f"  Salidas:  {config.n_salidas}\n"
            "El modelo está listo para simulación.\n"
            "Puede cargar los datos nuevamente si\n"
            "desea evaluar o reentrenar.\n"
        )
        self.status.set("Modelo cargado correctamente.", "ok")