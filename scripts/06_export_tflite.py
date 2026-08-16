#!/usr/bin/env python
"""Wrapper fino de CLI - chama handwash.application.export_tflite (Secao 7 do plano:
quantizacao INT8 full-integer, normalizacao embutida no grafo, dataset representativo
so do split de treino)."""
import json

from handwash.application.export_tflite import export_tflite_int8

if __name__ == "__main__":
    summary = export_tflite_int8()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
