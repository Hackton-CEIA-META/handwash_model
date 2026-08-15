#!/usr/bin/env python
"""Wrapper fino de CLI - chama handwash.application.viability_check (GATE)."""
import json

from handwash.application.viability_check import run_viability_check

if __name__ == "__main__":
    summary = run_viability_check()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["gate_passed"]:
        print("\nGATE: PASSOU - viabilidade confirmada (at_least_one_hand_rate), pode seguir pra extracao completa.")
    else:
        print("\nGATE: ATENCAO - pastas abaixo do limiar de at_least_one_hand_rate:", summary["folders_below_threshold"])
