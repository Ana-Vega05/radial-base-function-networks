import tkinter as tk
from tkinter import ttk
from styles import BG_CARD, ACCENT, ACCENT2, SUCCESS, WARNING, DANGER, FONT_SMALL

class PanelReporteClase(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG_CARD, **kwargs)
        self._build()

    def _build(self):
        frame = tk.Frame(self, bg=BG_CARD)
        frame.pack(fill="both", expand=True, padx=8, pady=4)

        vsb = ttk.Scrollbar(frame, orient="vertical")
        hsb = ttk.Scrollbar(frame, orient="horizontal")
        cols = ("Clase", "Precisión", "Recall", "F1", "Soporte")
        self.tree = ttk.Treeview(
            frame, columns=cols, show="headings", style="Dark.Treeview",
            yscrollcommand=vsb.set, xscrollcommand=hsb.set, height=8)
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)

        widths = {"Clase": 130, "Precisión": 80, "Recall": 75,
                  "F1": 75, "Soporte": 70}
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=widths.get(c, 80), anchor="center")
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

    def actualizar(self, res: dict):
        for row in self.tree.get_children():
            self.tree.delete(row)

        reporte_str = res.get("reporte", "")
        if not reporte_str:
            return
        lineas = reporte_str.strip().splitlines()
        colores = [ACCENT, ACCENT2, SUCCESS, WARNING, DANGER]
        clase_idx = 0
        for linea in lineas:
            linea = linea.strip()
            if not linea or linea.startswith("precision"):
                continue
            partes = linea.split()
            nums = []
            nombre_partes = []
            for p in partes:
                try:
                    nums.append(float(p))
                except ValueError:
                    nombre_partes.append(p)
            if len(nums) >= 4:
                nombre = " ".join(nombre_partes) if nombre_partes else f"Clase {clase_idx}"
                prec_val, rec_val, f1_val, soporte = nums[0], nums[1], nums[2], int(nums[3])
                tag = "avg" if any(k in nombre.lower() for k in ("avg", "accuracy", "macro", "weighted")) else f"c{clase_idx % len(colores)}"
                self.tree.insert("", "end", tags=(tag,), values=(
                    nombre,
                    f"{prec_val:.4f}",
                    f"{rec_val:.4f}",
                    f"{f1_val:.4f}",
                    str(soporte),
                ))
                clase_idx += 1

        for i, color in enumerate(colores):
            self.tree.tag_configure(f"c{i}", foreground=color)
        self.tree.tag_configure("avg", foreground=WARNING, font=("Segoe UI", 9, "bold"))