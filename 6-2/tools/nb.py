"""노트북 생성기 공통 — 셀을 모아 .ipynb 로 쓴다(출력 없이). build_notebook_0N.py 가 씀"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


class Book:
    def __init__(self, name):
        self.path = NB_DIR / name
        self.cells = []

    def md(self, s):
        self.cells.append({"cell_type": "markdown", "id": f"c{len(self.cells):02d}", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})

    def code(self, s):
        self.cells.append({"cell_type": "code", "id": f"c{len(self.cells):02d}", "metadata": {}, "execution_count": None, "outputs": [],
                           "source": s.strip("\n").splitlines(keepends=True)})

    def save(self):
        nb = {"cells": self.cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        self.path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(self.path, len(self.cells), "cells")
