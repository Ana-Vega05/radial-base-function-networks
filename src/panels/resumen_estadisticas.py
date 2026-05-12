import tkinter as tk
from styles import BG_PANEL, TEXT_DIM, ACCENT, ACCENT2, SUCCESS, WARNING, FONT_SMALL

class PanelResumenEstadisticas(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG_PANEL, **kwargs)
        self._sum_vars = {}
        self._build()

    def _build(self):
        labels = [
            ("n_patrones", "Patrones",  ACCENT),
            ("n_entradas", "Entradas",  ACCENT2),
            ("n_salidas",  "Salidas",   SUCCESS),
            ("n_clases",   "Clases",    WARNING),
        ]
        for i, (key, texto, color) in enumerate(labels):
            f = tk.Frame(self, bg=BG_PANEL, relief="flat")
            f.place(relx=i * 0.25 + 0.005, rely=0.05,
                    relwidth=0.24, relheight=0.90)
            tk.Label(f, text=texto, bg=BG_PANEL, fg=TEXT_DIM,font=FONT_SMALL).pack(pady=(8, 0))
            var = tk.StringVar(value="—")
            tk.Label(f, textvariable=var, bg=BG_PANEL, fg=color,font=("Segoe UI", 18, "bold")).pack()
            self._sum_vars[key] = var

    def actualizar(self, n_patrones, n_entradas, n_salidas, n_clases):
        self._sum_vars["n_patrones"].set(str(n_patrones))
        self._sum_vars["n_entradas"].set(str(n_entradas))
        self._sum_vars["n_salidas"].set(str(n_salidas))
        self._sum_vars["n_clases"].set(str(n_clases))