#!/usr/bin/env python
"""Wrapper fino de CLI - chama handwash.application.build_handoff_package (Secao 8 do
plano: monta classes.json + sample_io_pairs.npz em models/artifacts/, junto do .tflite
ja exportado por 06_export_tflite.py). MODEL_CARD.md e parity_check.py sao autorais,
ja escritos direto em models/artifacts/."""
import json

from handwash.application.build_handoff_package import build_handoff_package

if __name__ == "__main__":
    summary = build_handoff_package()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
