# -*- coding: utf-8 -*-
"""Panel izquierdo de Configuración – solo dataset."""

import os
import random
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from styles import (
    BG_CARD, TEXT_MAIN, TEXT_DIM, SUCCESS, FONT_BODY, FONT_SMALL, BG_PANEL,
    BG_DARK
)
from widgets import panel_btn, primary_btn, section_label, separator, action_btn

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


class PanelConfiguracion(tk.Frame):
    def __init__(self, parent, app_state: dict, on_cargar_datos, **kwargs):
        super().__init__(parent, bg=BG_CARD, **kwargs)
        self.app_state = app_state
        self._on_cargar_datos = on_cargar_datos
        self._current_path = None
        self._features = []
        self._n_inputs = 0
        self._n_outputs = 0
        self._n_records = 0

        self._build()

    def _build(self):
        section_label(self, "Carga del Dataset")
        separator(self)

        ds_row = tk.Frame(self, bg=BG_CARD)
        ds_row.pack(fill="x", padx=12, pady=4)

        tk.Label(ds_row, text="Archivo:", bg=BG_CARD, fg=TEXT_MAIN,font=FONT_BODY, anchor="w", width=8).pack(side="left")

        try:
            _raw_files = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(RAW_DATA_DIR)
                if f.endswith(".json")
            )
        except Exception:
            _raw_files = []

        self.ds_var = tk.StringVar()
        self.ds_combo = ttk.Combobox(
            ds_row, textvariable=self.ds_var,
            values=_raw_files, state="readonly", width=22, font=FONT_BODY)
        self.ds_combo.pack(side="left", padx=(4, 6))
        self.ds_combo.bind("<<ComboboxSelected>>", self._on_dataset_selected)

        action_btn(ds_row, "Examinar…", command=self._examinar,color=BG_DARK).pack(side="left")

        # Información detectada
        info_frame = tk.Frame(self, bg=BG_CARD)
        info_frame.pack(fill="x", padx=14, pady=(4, 0))

        self.lbl_entradas = tk.Label(
            info_frame, text="Entradas detectadas : —",
            bg=BG_CARD, fg=TEXT_DIM, font=FONT_SMALL, anchor="w")
        self.lbl_entradas.pack(anchor="w", pady=1)

        self.lbl_salidas = tk.Label(
            info_frame, text="Salidas detectadas  : —",
            bg=BG_CARD, fg=TEXT_DIM, font=FONT_SMALL, anchor="w")
        self.lbl_salidas.pack(anchor="w", pady=1)

        self.lbl_patrones = tk.Label(
            info_frame, text="Patrones totales  : —",
            bg=BG_CARD, fg=TEXT_DIM, font=FONT_SMALL, anchor="w")
        self.lbl_patrones.pack(anchor="w", pady=(1, 6))

        separator(self)

        btn_row = tk.Frame(self, bg=BG_CARD)
        btn_row.pack(fill="x", padx=12, pady=10)

        self.btn_cargar = primary_btn(btn_row, "📂  Cargar datos",self._on_cargar_datos)
        self.btn_cargar.pack(side="left", padx=(0, 10))

        self.btn_aleatorio = panel_btn(btn_row, "🎲  Selección aleatoria",self._seleccion_aleatoria)
        self.btn_aleatorio.pack(side="left")

        self.btn_cargar_modelo = panel_btn(btn_row, "📥 Cargar modelo", self._on_cargar_modelo)
        self.btn_cargar_modelo.pack(side="left", padx=(10, 0))

    # Métodos públicos
    def set_buttons_state(self, state: str):
        self.btn_aleatorio.configure(state=state)
        self.btn_cargar.configure(state=state)

    def get_current_path(self):
        return self._current_path

    def get_features(self):
        return self._features

    def get_n_inputs(self):
        return self._n_inputs

    def get_n_outputs(self):
        return self._n_outputs

    def get_n_records(self):
        return self._n_records

    def set_cargar_modelo_callback(self, callback):
        self._cargar_modelo_callback = callback

    def _on_cargar_modelo(self):
        if hasattr(self, '_cargar_modelo_callback'):
            self._cargar_modelo_callback()

    # Eventos internos 
    def _on_dataset_selected(self, _event=None):
        name = self.ds_var.get()
        if not name:
            return
        path = os.path.join(RAW_DATA_DIR, f"{name}.json")
        self._set_dataset(path)

    def _examinar(self):
        path = filedialog.askopenfilename(
            title="Seleccionar dataset JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not path:
            return
        self.ds_var.set(os.path.splitext(os.path.basename(path))[0])
        self._set_dataset(path)

    def _set_dataset(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            data = d.get("data", d) if isinstance(d, dict) else d
            features = d.get("features", []) if isinstance(d, dict) else []
            if not data:
                raise ValueError("Dataset vacío")
            first = data[0]
            if "input" in first:
                n_in = len(first["input"])
                n_out = 1
                if not features:
                    features = [f"x{i+1}" for i in range(n_in)]
            else:
                cols = list(first.keys())
                n_in = len(cols) - 1
                n_out = 1
                if not features:
                    features = cols[:-1]
        except Exception as e:
            messagebox.showerror("Error al leer dataset", str(e))
            return

        self._current_path = path
        self._features = features
        self._n_inputs = n_in
        self._n_outputs = n_out
        self._n_records = len(data)

        feat_str = ", ".join(features) if features else "—"
        self.lbl_entradas.configure(
            text=f"Entradas detectadas : {n_in}  ({feat_str})",
            fg=SUCCESS)
        self.lbl_salidas.configure(
            text=f"Salidas detectadas  : {n_out}", fg=SUCCESS)
        self.lbl_patrones.configure(
            text=f"Patrones totales  : {len(data)}", fg=SUCCESS)

    def _seleccion_aleatoria(self):
        """Selecciona un dataset aleatorio si hay disponibles."""
        if not self.ds_combo['values']:
            messagebox.showwarning("Sin datasets","No hay datasets en la carpeta raw/")
            return
        import random
        name = random.choice(self.ds_combo['values'])
        self.ds_var.set(name)
        self._on_dataset_selected()