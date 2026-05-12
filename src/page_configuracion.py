# -*- coding: utf-8 -*-
"""Vista de Configuración — carga de dataset, parámetros RBF y resumen."""

import os
import json
import random
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from styles import (
    BG_DARK, BG_CARD, BG_PANEL, ACCENT, ACCENT2, TEXT_MAIN, TEXT_DIM,
    SUCCESS, WARNING, DANGER, BORDER, FONT_TITLE, FONT_HEADER,
    FONT_BODY, FONT_SMALL, FONT_MONO,
)
from widgets import card, section_label, separator, labeled_entry, action_btn

_SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
RAW_DATA_DIR  = os.path.join(_PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")

from core.config        import Config
from core.data_loader   import load_json_dataset
from core.preprocessing import split_dataset
from core.trainer       import entrenar_rbf
from core.evaluator     import evaluar_modelo



def _peek_dataset(json_path: str):
    """Devuelve (features, n_inputs, n_outputs, n_records) sin Config."""
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    data     = d.get("data", d) if isinstance(d, dict) else d
    features = d.get("features", []) if isinstance(d, dict) else []
    if not data:
        return [], 0, 0, 0
    first = data[0]
    if "input" in first:
        n_in  = len(first["input"])
        n_out = 1
        if not features:
            features = [f"x{i+1}" for i in range(n_in)]
    else:
        cols  = list(first.keys())
        n_in  = len(cols) - 1
        n_out = 1
        if not features:
            features = cols[:-1]
    return features, n_in, n_out, len(data)



class PageConfiguracion(tk.Frame):
    def __init__(self, parent, status_bar, app_state: dict):
        super().__init__(parent, bg=BG_DARK)
        self.status    = status_bar
        self.app_state = app_state
        self._current_path: str | None = None  # ruta JSON activa
        self._features: list  = []
        self._n_inputs: int   = 0
        self._n_outputs: int  = 0
        self._n_records: int  = 0
        self._running: bool   = False
        self._build()


    def _build(self):
        tk.Label(self, text="Configuración de la Red RBF", bg=BG_DARK,
                 fg=TEXT_MAIN, font=FONT_TITLE).pack(
                     anchor="w", padx=24, pady=(20, 2))
        tk.Label(self,
                 text="Seleccione el dataset y defina los parámetros de la red antes de ejecutar.",
                 bg=BG_DARK, fg=TEXT_DIM, font=FONT_SMALL).pack(
                     anchor="w", padx=24)

        left = card(self)
        left.place(relx=0.01, rely=0.12, relwidth=0.45, relheight=0.86)

        # 4.1 Dataset
        section_label(left, "4.1  Carga del Dataset")
        separator(left)

        ds_row = tk.Frame(left, bg=BG_CARD)
        ds_row.pack(fill="x", padx=12, pady=4)

        tk.Label(ds_row, text="Archivo:", bg=BG_CARD, fg=TEXT_MAIN,
                 font=FONT_BODY, anchor="w", width=8).pack(side="left")

        # Poblar combo con los 4 datasets del raw/
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

        action_btn(ds_row, "Examinar…", command=self._examinar,
                   color=BG_DARK).pack(side="left")

        # 4.2 / 4.3 — info detectada
        info_frame = tk.Frame(left, bg=BG_CARD)
        info_frame.pack(fill="x", padx=14, pady=(4, 0))

        self.lbl_entradas = tk.Label(
            info_frame, text="4.2  Entradas detectadas : —",
            bg=BG_CARD, fg=TEXT_DIM, font=FONT_SMALL, anchor="w")
        self.lbl_entradas.pack(anchor="w", pady=1)

        self.lbl_salidas = tk.Label(
            info_frame, text="4.3  Salidas detectadas  : —",
            bg=BG_CARD, fg=TEXT_DIM, font=FONT_SMALL, anchor="w")
        self.lbl_salidas.pack(anchor="w", pady=1)

        self.lbl_patrones = tk.Label(
            info_frame, text="       Patrones totales  : —",
            bg=BG_CARD, fg=TEXT_DIM, font=FONT_SMALL, anchor="w")
        self.lbl_patrones.pack(anchor="w", pady=(1, 6))

        separator(left)

        # 4.4 – 4.7 Parámetros
        section_label(left, "Parámetros de la Red")
        separator(left)

        self.var_centros   = labeled_entry(left, "4.4  Neuronas / centros:", "", width=10)
        self.var_error     = labeled_entry(left, "4.5  Error aprox. (0–0.1):", "", width=10)
        self.var_iter      = labeled_entry(left, "4.6  Iteraciones (mín 1):", "", width=10)

        # 4.7 Esquema de partición
        part_row = tk.Frame(left, bg=BG_CARD)
        part_row.pack(fill="x", padx=12, pady=3)
        tk.Label(part_row, text="4.7  Esquema partición:", bg=BG_CARD,
                 fg=TEXT_MAIN, font=FONT_BODY, width=22, anchor="w").pack(side="left")
        self.var_particion = tk.StringVar(value="70-15-15")
        ttk.Combobox(part_row, textvariable=self.var_particion,
                     values=["70-15-15", "80-10-10"],
                     state="readonly", width=10, font=FONT_BODY).pack(
                         side="left", padx=(4, 0))

        separator(left)

        # Botones
        btn_row = tk.Frame(left, bg=BG_CARD)
        btn_row.pack(fill="x", padx=12, pady=10)

        self.btn_aleatorio = action_btn(
            btn_row, "🎲  Elección aleatoria",
            command=self._eleccion_aleatoria, color=BG_PANEL, fg=TEXT_MAIN)
        self.btn_aleatorio.pack(side="left")

        self.btn_cargar = action_btn(
            btn_row, "📂  Cargar datos",
            command=self._cargar_datos, color=ACCENT)
        self.btn_cargar.pack(side="left", padx=(10, 0))

        # ── Panel derecho ─────────────────────────────────────────────────────
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
        scroll = ttk.Scrollbar(text_frame, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.summary_text.pack(side="left", fill="both", expand=True)

        self._show_summary("Seleccione un dataset y configure los parámetros.\n\n"
                           "Haga clic en 'Elección aleatoria' para valores de ejemplo,\n"
                           "o ingrese sus propios valores y pulse 'Cargar datos'.")

    # ── Eventos de dataset ────────────────────────────────────────────────────

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
        # Mostrar nombre en combo (solo el basename sin ext)
        self.ds_var.set(os.path.splitext(os.path.basename(path))[0])
        self._set_dataset(path)

    def _set_dataset(self, path: str):
        try:
            features, n_in, n_out, n_rec = _peek_dataset(path)
        except Exception as e:
            messagebox.showerror("Error al leer dataset", str(e))
            return
        self._current_path = path
        self._features  = features
        self._n_inputs  = n_in
        self._n_outputs = n_out
        self._n_records = n_rec
        feat_str = ", ".join(features) if features else "—"
        self.lbl_entradas.configure(
            text=f"4.2  Entradas detectadas : {n_in}  ({feat_str})",
            fg=SUCCESS)
        self.lbl_salidas.configure(
            text=f"4.3  Salidas detectadas  : {n_out}", fg=SUCCESS)
        self.lbl_patrones.configure(
            text=f"       Patrones totales  : {n_rec}", fg=SUCCESS)
        self.status.set(f"Dataset '{os.path.basename(path)}' detectado — {n_in} entradas.", "warn")

    # ── Elección aleatoria ────────────────────────────────────────────────────

    def _eleccion_aleatoria(self):
        if not self._current_path:
            messagebox.showwarning(
                "Dataset requerido",
                "Seleccione un dataset primero para generar valores aleatorios.")
            return

        n_in    = self._n_inputs or 1
        centros = random.randint(n_in, max(n_in * 3, n_in + 5))
        error   = round(random.uniform(0.001, 0.05), 4)
        iters   = random.randint(10, 100)
        particion = random.choice(["70-15-15", "80-10-10"])

        self.var_centros.set(str(centros))
        self.var_error.set(str(error))
        self.var_iter.set(str(iters))
        self.var_particion.set(particion)

        nombre = os.path.splitext(os.path.basename(self._current_path))[0]
        feat_str = ", ".join(self._features) if self._features else "—"

        resumen = (
            "═══════════════════════════════════════\n"
            "  VALORES ALEATORIOS ASIGNADOS\n"
            "═══════════════════════════════════════\n\n"
            f"  Dataset         : {nombre}\n"
            f"  Entradas ({self._n_inputs})    : {feat_str}\n"
            f"  Salidas         : {self._n_outputs}\n"
            f"  Patrones        : {self._n_records}\n\n"
            f"  Neuronas/centros: {centros}\n"
            f"  Error óptimo    : {error}\n"
            f"  Iteraciones máx : {iters}\n"
            f"  Partición       : {particion}\n\n"
            "───────────────────────────────────────\n"
            "  Presione 'Cargar datos' para ejecutar\n"
            "  el pipeline completo.\n"
        )
        self._show_summary(resumen)
        self.status.set("Valores aleatorios asignados. Presione 'Cargar datos'.", "warn")

    # ── Cargar datos ──────────────────────────────────────────────────────────

    def _cargar_datos(self):
        if self._running:
            return
        if not self._current_path:
            messagebox.showerror("Error", "Seleccione un dataset primero.")
            return

        # Validar campos
        try:
            centros = int(self.var_centros.get())
            assert centros >= 1
        except Exception:
            messagebox.showerror("Error", "Neuronas/centros debe ser un entero ≥ 1.")
            return

        try:
            error = float(self.var_error.get())
            assert 0.0 <= error <= 0.1
        except Exception:
            messagebox.showerror("Error", "Error de aproximación debe estar entre 0.0 y 0.1.")
            return

        try:
            iters = int(self.var_iter.get())
            assert iters >= 1
        except Exception:
            messagebox.showerror("Error", "Iteraciones debe ser un entero ≥ 1.")
            return

        if self._n_inputs > 0 and centros < self._n_inputs:
            messagebox.showerror(
                "Error",
                f"Neuronas/centros ({centros}) debe ser ≥ n_entradas ({self._n_inputs}).")
            return

        particion  = self.var_particion.get()
        nombre     = os.path.splitext(os.path.basename(self._current_path))[0]
        raw_dir    = os.path.dirname(self._current_path)
        features   = self._features or [f"x{i+1}" for i in range(self._n_inputs)]

        config_dict = {
            "raw_data_path":       raw_dir,
            "processed_data_path": PROCESSED_DIR,
            "dataset_name":        nombre,
            "input_columns":       features,
            "target_column":       "y",
            "n_centros":           centros,
            "error_optimo":        error,
            "max_iteraciones":     iters,
            "tipo_particion":      particion,
            "random_state":        None,
        }

        self._running = True
        self._set_buttons_state("disabled")
        self._show_summary("⏳  Ejecutando pipeline...\n\n"
                           "  · Cargando dataset\n"
                           "  · Particionando\n"
                           "  · Entrenando red RBF\n"
                           "  · Evaluando modelo\n\n"
                           "Por favor espere.")
        self.status.set("Ejecutando pipeline… esto puede tomar unos segundos.", "warn")

        t = threading.Thread(
            target=self._run_pipeline_thread,
            args=(config_dict,), daemon=True)
        t.start()

    def _run_pipeline_thread(self, config_dict: dict):
        try:
            config = Config.from_dict(config_dict)
            X, Yd, info = load_json_dataset(config)
            splits = split_dataset(X, Yd, config.tipo_particion, config.random_state)
            modelo, hist_EG, hist_centros, convergio = entrenar_rbf(
                splits["X_train"], splits["Yd_train"], config,
                info["min_X"], info["max_X"], verbose=False)
            resultados = evaluar_modelo(
                modelo, splits["X_test"], splits["Yd_test"],
                verbose=False, diagnostico=False)

            payload = {
                "config":            config,
                "X":                 X,
                "Yd":                Yd,
                "info":              info,
                "splits":            splits,
                "modelo":            modelo,
                "resultados":        resultados,
                "historial_EG":      hist_EG,
                "historial_centros": hist_centros,
                "convergio":         convergio,
            }
            self.after(0, lambda: self._on_done(payload))
        except Exception as exc:
            msg = str(exc)
            self.after(0, lambda: self._on_error(msg))

    def _on_done(self, payload: dict):
        self.app_state.update(payload)
        self._running = False
        self._set_buttons_state("normal")

        cfg  = payload["config"]
        info = payload["info"]
        res  = payload["resultados"]
        hist = payload["historial_EG"]
        conv = payload["convergio"]
        feat_str = ", ".join(cfg.input_columns)

        resumen = (
            "═══════════════════════════════════════\n"
            "  RESUMEN DE CONFIGURACIÓN Y RESULTADOS\n"
            "═══════════════════════════════════════\n\n"
            f"  Dataset         : {cfg.dataset_name}\n"
            f"  Entradas ({info['n_entradas']})   : {feat_str}\n"
            f"  Salidas         : {info['n_salidas']}\n"
            f"  Patrones totales: {info['n_patrones']}\n\n"
            "  ── Parámetros configurados ──\n"
            f"  Neuronas/centros: {cfg.n_centros}\n"
            f"  Error óptimo    : {cfg.error_optimo}\n"
            f"  Iteraciones máx : {cfg.max_iteraciones}\n"
            f"  Partición       : {cfg.tipo_particion}\n\n"
            "  ── Resultado del entrenamiento ──\n"
            f"  Estado          : {'CONVERGE' if conv else 'NO CONVERGE'}\n"
            f"  Centros finales : {payload['modelo'].centros.shape[0]}\n"
            f"  Iteraciones     : {len(hist)}\n"
            f"  EG entrenamiento: {hist[-1]:.6f}\n\n"
            "  ── Métricas de evaluación ──\n"
            f"  EG prueba (MAE) : {res['EG_test']:.6f}\n"
            f"  Exactitud       : {res['exactitud']*100:.2f}%\n"
            f"  Precisión       : {res['precision']:.4f}\n"
            f"  Sensibilidad    : {res['sensibilidad']:.4f}\n"
            f"  F1-Score        : {res['f1']:.4f}\n"
        )
        self._show_summary(resumen)
        self.status.set(
            f"Pipeline completo — Exactitud: {res['exactitud']*100:.2f}%  "
            f"({'CONVERGE' if conv else 'NO CONVERGE'})", "ok")

    def _on_error(self, msg: str):
        self._running = False
        self._set_buttons_state("normal")
        self._show_summary(f"❌  Error durante el pipeline:\n\n{msg}")
        self.status.set(f"Error: {msg[:80]}", "error")
        messagebox.showerror("Error en el pipeline", msg)

    # ── Utilidades ────────────────────────────────────────────────────────────

    def _show_summary(self, text: str):
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("end", text)
        self.summary_text.configure(state="disabled")

    def _set_buttons_state(self, state: str):
        self.btn_aleatorio.configure(state=state)
        self.btn_cargar.configure(state=state)
