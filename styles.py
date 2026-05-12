# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk

BG_DARK   = "#1e2130"
BG_PANEL  = "#252a3d"
BG_CARD   = "#2d3250"
ACCENT    = "#7c83d6"
ACCENT2   = "#56c8d8"
TEXT_MAIN = "#e8eaf6"
TEXT_DIM  = "#8890b5"
SUCCESS   = "#4caf80"
WARNING   = "#f0a500"
DANGER    = "#e05c5c"
BORDER    = "#3a4060"

FONT_TITLE  = ("Segoe UI", 13, "bold")
FONT_HEADER = ("Segoe UI", 11, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 9)


def apply_ttk_styles(root: tk.Tk) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure("Dark.Treeview",
                    background=BG_DARK, foreground=TEXT_MAIN,
                    fieldbackground=BG_DARK, rowheight=22)
    style.configure("Dark.Treeview.Heading",
                    background=BG_PANEL, foreground=ACCENT)
    style.map("Dark.Treeview", background=[("selected", ACCENT)])

    style.configure("TCombobox",
                    fieldbackground=BG_DARK, background=BG_DARK,
                    foreground=TEXT_MAIN, arrowcolor=ACCENT,
                    selectbackground=BG_DARK, selectforeground=TEXT_MAIN)
    style.map("TCombobox",
              fieldbackground=[("readonly", BG_DARK)],
              foreground=[("readonly", TEXT_MAIN)])

    style.configure("TProgressbar",
                    troughcolor=BG_DARK, background=ACCENT, thickness=10)
