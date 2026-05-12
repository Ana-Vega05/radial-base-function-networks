# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
from styles import (
    BG_DARK, BG_PANEL, BG_CARD, ACCENT, ACCENT2,
    TEXT_MAIN, TEXT_DIM, SUCCESS, WARNING, DANGER, BORDER,
    FONT_HEADER, FONT_BODY, FONT_SMALL,
)

def card(parent, **kwargs) -> tk.Frame:
    return tk.Frame(parent, bg=BG_CARD, relief="flat", bd=0, **kwargs)


def section_label(parent, text: str) -> None:
    tk.Label(parent, text=text, bg=BG_CARD, fg=ACCENT,
             font=FONT_HEADER).pack(anchor="w", padx=12, pady=(12, 4))


def separator(parent) -> None:
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=2)


def labeled_entry(parent, label_text: str, default: str = "",
                  width: int = 18) -> tk.StringVar:
    row = tk.Frame(parent, bg=BG_CARD)
    row.pack(fill="x", padx=12, pady=3)
    tk.Label(row, text=label_text, bg=BG_CARD, fg=TEXT_MAIN,
             font=FONT_BODY, width=22, anchor="w").pack(side="left")
    var = tk.StringVar(value=default)
    tk.Entry(row, textvariable=var, bg=BG_DARK, fg=TEXT_MAIN,
             insertbackground=TEXT_MAIN, relief="flat", bd=4,
             font=FONT_BODY, width=width).pack(side="left", padx=(4, 0))
    return var


def labeled_combo(parent, label_text: str, values: list,
                  default: int = 0) -> tk.StringVar:
    row = tk.Frame(parent, bg=BG_CARD)
    row.pack(fill="x", padx=12, pady=3)
    tk.Label(row, text=label_text, bg=BG_CARD, fg=TEXT_MAIN,
             font=FONT_BODY, width=22, anchor="w").pack(side="left")
    var = tk.StringVar(value=values[default])
    ttk.Combobox(row, textvariable=var, values=values,
                 state="readonly", width=16, font=FONT_BODY).pack(
                     side="left", padx=(4, 0))
    return var


def action_btn(parent, text: str, command=None,
               color: str = ACCENT, fg: str = TEXT_MAIN) -> tk.Button:
    return tk.Button(
        parent, text=text, command=command,
        bg=color, fg=fg, activebackground=ACCENT2,
        activeforeground=BG_DARK, relief="flat", bd=0,
        font=("Segoe UI", 10, "bold"), cursor="hand2",
        padx=16, pady=6,
    )

def primary_btn(parent, text: str, command=None) -> tk.Button:
    """Botón con color de acento principal (azul)."""
    return action_btn(parent, text, command, color=ACCENT)

def success_btn(parent, text: str, command=None) -> tk.Button:
    """Botón para acciones de éxito (verde)."""
    return action_btn(parent, text, command, color=SUCCESS)

def danger_btn(parent, text: str, command=None) -> tk.Button:
    """Botón para acciones peligrosas (rojo)."""
    return action_btn(parent, text, command, color=DANGER)

def panel_btn(parent, text: str, command=None) -> tk.Button:
    """Botón estilo panel oscuro (usado en elección aleatoria, etc.)."""
    return action_btn(parent, text, command, color=BG_PANEL, fg=TEXT_MAIN)

def dark_btn(parent, text: str, command=None) -> tk.Button:
    """Botón con fondo oscuro, usado para acciones secundarias."""
    return action_btn(parent, text, command, color=BG_DARK, fg=TEXT_MAIN)


class Sidebar(tk.Frame):
    ITEMS = [
        ("⚙",  "Configuración",  0),
        ("📊", "Estadísticas",   1),
        ("🎯", "Entrenamiento",  2),
        ("▶",  "Simulación",    3),
        ("📈", "Evaluación",     4),
    ]

    def __init__(self, parent, on_select):
        super().__init__(parent, bg=BG_PANEL, width=190)
        self.pack_propagate(False)
        self.on_select = on_select
        self.buttons: dict = {}
        self._build_logo()
        for icon, label, idx in self.ITEMS:
            self._add_item(icon, label, idx)

    def _build_logo(self) -> None:
        tk.Label(self, text="RBF Network", bg=BG_PANEL, fg=ACCENT,
                 font=("Segoe UI", 14, "bold")).pack(pady=(20, 4))
        tk.Label(self, text="Red de Base Radial", bg=BG_PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL).pack()
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=16, pady=14)

    def _add_item(self, icon: str, label: str, idx: int) -> None:
        btn = tk.Button(
            self, text=f"  {icon}  {label}",
            bg=BG_PANEL, fg=TEXT_DIM, relief="flat", bd=0,
            font=FONT_BODY, anchor="w", cursor="hand2",
            activebackground=BG_CARD, activeforeground=TEXT_MAIN,
            command=lambda i=idx: self._select(i),
        )
        btn.pack(fill="x", padx=8, pady=1, ipady=8)
        self.buttons[idx] = btn

    def _select(self, idx: int) -> None:
        for i, btn in self.buttons.items():
            btn.configure(
                bg=BG_CARD if i == idx else BG_PANEL,
                fg=TEXT_MAIN if i == idx else TEXT_DIM,
                font=("Segoe UI", 10, "bold") if i == idx else FONT_BODY,
            )
        self.on_select(idx)

    def select(self, idx: int) -> None:
        self._select(idx)



class StatusBar(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG_PANEL, height=28)
        self._var = tk.StringVar(value="Listo.")
        tk.Label(self, textvariable=self._var, bg=BG_PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL, anchor="w").pack(side="left", padx=12)
        self._state_lbl = tk.Label(self, text="● Sin datos",
                                   bg=BG_PANEL, fg=DANGER, font=FONT_SMALL)
        self._state_lbl.pack(side="right", padx=12)

    def set(self, msg: str, state: str = None) -> None:
        self._var.set(msg)
        if state == "ok":
            self._state_lbl.configure(text="● Activo",      fg=SUCCESS)
        elif state == "warn":
            self._state_lbl.configure(text="● Advertencia", fg=WARNING)
        elif state == "error":
            self._state_lbl.configure(text="● Error",       fg=DANGER)
