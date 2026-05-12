import sys
import os

import matplotlib
matplotlib.use("TkAgg")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_SRC  = os.path.join(PROJECT_ROOT, "src")

for _p in [PROJECT_ROOT, PROJECT_SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tkinter as tk
from tkinter import ttk

from styles import apply_ttk_styles, BG_DARK
from widgets import Sidebar, StatusBar
from src.page_configuracion import PageConfiguracion
from src.page_estadisticas   import PageEstadisticas
from src.page_entrenamiento  import PageEntrenamiento
from src.page_simulacion     import PageSimulacion
from src.page_evaluacion     import PageEvaluacion

class RBFApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Red de Base Radial — RBF Network")
        self.geometry("1200x720")
        self.minsize(1000, 620)
        self.configure(bg=BG_DARK)
        apply_ttk_styles(self)

        # Estado compartido entre páginas
        self.app_state: dict = {
            "dataset_path":      None,
            "config":            None,
            "X":                 None,
            "Yd":                None,
            "info":              None,
            "splits":            None,
            "modelo":            None,
            "resultados":        None,
            "historial_EG":      [],
            "historial_centros": [],
            "convergio":         False,
        }

        self._build()

    def _build(self):
        self.status_bar = StatusBar(self)
        self.status_bar.pack(side="bottom", fill="x")

        self.sidebar = Sidebar(self, self._show_page)
        self.sidebar.pack(side="left", fill="y")

        self.container = tk.Frame(self, bg=BG_DARK)
        self.container.pack(side="right", fill="both", expand=True)

        # Páginas con app_state (configuracion, estadisticas, evaluacion)
        # Páginas sin app_state (entrenamiento, simulacion — no se tocan)
        page_classes = [
            PageConfiguracion,
            PageEstadisticas,
            PageEntrenamiento,
            PageSimulacion,
            PageEvaluacion,
        ]
        uses_state = {0, 1, 4}

        self.pages = []
        for i, Cls in enumerate(page_classes):
            if i in uses_state:
                page = Cls(self.container, self.status_bar, self.app_state)
            else:
                page = Cls(self.container, self.status_bar)
            page.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.pages.append(page)

        self._show_page(0)
        self.sidebar.select(0)

    def _show_page(self, idx):
        self.pages[idx].lift()
        if hasattr(self.pages[idx], "on_show"):
            self.pages[idx].on_show()


if __name__ == "__main__":
    app = RBFApp()
    app.mainloop()
