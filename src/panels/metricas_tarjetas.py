import tkinter as tk
from styles import BG_PANEL, TEXT_DIM, ACCENT, ACCENT2, SUCCESS, WARNING, DANGER, FONT_SMALL

class PanelMetricasTarjetas(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG_PANEL, **kwargs)
        self._metric_vars = {}
        self._build()

    def _build(self):
        definitions = [
            ("exactitud",    "Exactitud",            ACCENT),
            ("precision",    "Precisión",            ACCENT2),
            ("sensibilidad", "Sensibilidad",         SUCCESS),
            ("f1",           "F1-Score",             WARNING),
            ("eg_test",      "EG Prueba",            TEXT_DIM),
            ("estado",       "Convergencia",         TEXT_DIM),
        ]
        n = len(definitions)
        w = 1.0 / n
        for i, (key, etiqueta, color) in enumerate(definitions):
            f = tk.Frame(self, bg=BG_PANEL, relief="flat")
            f.place(relx=i * w + 0.003, rely=0.05,
                    relwidth=w - 0.006, relheight=0.90)
            tk.Label(f, text=etiqueta, bg=BG_PANEL, fg=TEXT_DIM,font=FONT_SMALL).pack(pady=(7, 0))
            var = tk.StringVar(value="—")
            self._metric_vars[key] = var
            tk.Label(f, textvariable=var, bg=BG_PANEL, fg=color,font=("Segoe UI", 15, "bold")).pack()

    def actualizar(self, res, convergio):
        self._metric_vars["exactitud"].set(f"{res['exactitud']*100:.2f}%")
        self._metric_vars["precision"].set(f"{res['precision']:.4f}")
        self._metric_vars["sensibilidad"].set(f"{res['sensibilidad']:.4f}")
        self._metric_vars["f1"].set(f"{res['f1']:.4f}")
        self._metric_vars["eg_test"].set(f"{res['EG_test']:.5f}")

        estado_txt = "CONVERGE" if convergio else "NO CONVERGE"
        self._metric_vars["estado"].set(estado_txt)
        # Cambiar color dinámico (se hará desde la página principal si se desea)