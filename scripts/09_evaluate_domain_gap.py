#!/usr/bin/env python
"""Wrapper fino de CLI - chama handwash.application.evaluate_domain_gap (Dia 3: domain
gap camera fixa de pia -> camera de celular). Le data/custom/manifest.csv, extrai
landmarks dos videos listados, avalia contra os modelos ja treinados (Keras + TFLite,
sem retreino), escreve models/checkpoints/domain_gap_report.json."""
import json

from handwash.application.evaluate_domain_gap import evaluate_domain_gap

if __name__ == "__main__":
    summary = evaluate_domain_gap()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
