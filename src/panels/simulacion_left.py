# -*- coding: utf-8 -*-
"""Panel izquierdo de Simulación – modo, selector de patrón y entradas."""

import tkinter as tk
from tkinter import ttk

from styles import (
    BG_CARD, BG_DARK, TEXT_MAIN, TEXT_DIM, SUCCESS, DANGER, FONT_BODY
)
from widgets import (
    section_label, separator, labeled_combo, dark_btn, success_btn, danger_btn
)


class PanelSimulacionLeft(tk.Frame):
    def __init__(self, parent, app_state: dict,
                 on_simular, on_limpiar, **kwargs):
        super().__init__(parent, bg=BG_CARD, **kwargs)
        self.app_state = app_state
        self._on_simular = on_simular
        self._on_limpiar = on_limpiar

        # Widgets que la página necesita manipular desde fuera
        self.mode_var = None
        self._patron_spin = None
        self._load_patron_btn = None
        self.sim_entries: dict[str, tk.StringVar] = {}
        self._entries_frame = None

        self._build()

    def _build(self):
        section_label(self, "Fuente de Datos")
        separator(self)

        self.mode_var = labeled_combo(
            self, "Modo:",
            ["Entrada manual", "Conjunto de prueba", "Conjunto de validación"], 0)

        # Fila del selector de patrón (oculta al inicio)
        self._patron_row = tk.Frame(self, bg=BG_CARD)
        self._patron_row.pack(fill="x", padx=12, pady=3)
        tk.Label(self._patron_row, text="Patrón #:", bg=BG_CARD, fg=TEXT_MAIN,font=FONT_BODY, width=22, anchor="w").pack(side="left")
        self._patron_spin = tk.Spinbox(
            self._patron_row, from_=1, to=9999, width=8,
            bg=BG_DARK, fg=TEXT_MAIN, insertbackground=TEXT_MAIN,
            relief="flat", bd=4, font=FONT_BODY)
        self._patron_spin.pack(side="left", padx=(4, 0))
        self._patron_row.pack_forget()

        # Botón "Cargar patrón"
        self._load_patron_btn = dark_btn(self, "⬆ Cargar patrón", command=None)
        self._load_patron_btn.pack(anchor="w", padx=12, pady=2)
        self._load_patron_btn.pack_forget()

        separator(self)
        section_label(self, "Valores de Entrada")
        separator(self)

        # Marco para las entradas dinámicas
        self._entries_frame = tk.Frame(self, bg=BG_CARD)
        self._entries_frame.pack(fill="x")
        self.build_entries()  # inicial según configuración actual

        separator(self)
        btn_row = tk.Frame(self, bg=BG_CARD)
        btn_row.pack(fill="x", padx=12, pady=10)
        success_btn(btn_row, "▶  Simular", self._on_simular).pack(side="left")
        danger_btn(btn_row, "✖  Limpiar", self._on_limpiar).pack(side="left", padx=(8, 0))

    # Métodos públicos para la página
    def build_entries(self):
        """Recrea las filas de entrada según las columnas del config."""
        for w in self._entries_frame.winfo_children():
            w.destroy()
        self.sim_entries.clear()

        cfg = self.app_state.get("config")
        cols = list(cfg.input_columns) if cfg else [f"x{i}" for i in range(4)]

        for col in cols:
            row = tk.Frame(self._entries_frame, bg=BG_CARD)
            row.pack(fill="x", padx=12, pady=2)
            tk.Label(row, text=f"{col}:", bg=BG_CARD, fg=TEXT_MAIN,font=FONT_BODY, width=20, anchor="w").pack(side="left")
            var = tk.StringVar(value="0.0")
            tk.Entry(row, textvariable=var, bg=BG_DARK, fg=TEXT_MAIN,insertbackground=TEXT_MAIN, relief="flat", bd=4,font=FONT_BODY, width=14).pack(side="left", padx=(4, 0))
            self.sim_entries[col] = var

    def set_spin_max(self, max_val: int):
        self._patron_spin.config(to=max(max_val, 1))

    def get_spin_value(self) -> int:
        try:
            return int(self._patron_spin.get())
        except (ValueError, tk.TclError):
            return 1

    def set_entries_from_patron(self, patron, col_names):
        """Rellena las entradas con un vector numérico."""
        for j, col in enumerate(col_names):
            if col in self.sim_entries and j < len(patron):
                self.sim_entries[col].set(f"{patron[j]:.6f}")

    def get_entry_values(self) -> list:
        """Devuelve los valores de entrada en el orden de las columnas."""
        cfg = self.app_state.get("config")
        cols = list(cfg.input_columns) if cfg else list(self.sim_entries.keys())
        return [float(self.sim_entries[c].get()) for c in cols if c in self.sim_entries]

    def set_load_patron_command(self, cmd):
        self._load_patron_btn.configure(command=cmd)

    def show_pattern_controls(self, show: bool):
        if show:
            self._patron_row.pack(fill="x", padx=12, pady=3)
            self._load_patron_btn.pack(anchor="w", padx=12, pady=2)
        else:
            self._patron_row.pack_forget()
            self._load_patron_btn.pack_forget()

    def clear_entries(self):
        for var in self.sim_entries.values():
            var.set("0.0")