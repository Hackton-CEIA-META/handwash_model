#!/usr/bin/env python
"""Wrapper fino de CLI - chama handwash.application.evaluate_tflite_vs_keras (Secao 7.3
do plano: Keras FP32 vs TFLite INT8 sobre as mesmas janelas de teste, gate de ~2pp)."""
import json

from handwash.application.evaluate_tflite_vs_keras import evaluate_tflite_vs_keras

if __name__ == "__main__":
    summary = evaluate_tflite_vs_keras()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
