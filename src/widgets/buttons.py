import tkinter as tk
from styles import BG_PANEL, TEXT_MAIN, SUCCESS, DANGER, ACCENT

def action_btn(parent, text, command, color=BG_PANEL, fg=TEXT_MAIN, **kwargs):
    """Base genérica, mantenida por compatibilidad."""
    return tk.Button(
        parent, text=text, command=command,
        bg=color, fg=fg,
        font=("Segoe UI", 10, "bold"),
        relief="flat", padx=14, pady=5,
        **kwargs
    )

def primary_btn(parent, text, command, **kwargs):
    """Botón con color de acento (azul)."""
    return action_btn(parent, text, command, color=ACCENT, **kwargs)

def success_btn(parent, text, command, **kwargs):
    """Botón para acciones positivas (verde)."""
    return action_btn(parent, text, command, color=SUCCESS, **kwargs)

def danger_btn(parent, text, command, **kwargs):
    """Botón para acciones destructivas (rojo)."""
    return action_btn(parent, text, command, color=DANGER, **kwargs)

def panel_btn(parent, text, command, **kwargs):
    """Botón estilo panel oscuro (usado para 'Elección aleatoria')."""
    return action_btn(parent, text, command, color=BG_PANEL, fg=TEXT_MAIN, **kwargs)