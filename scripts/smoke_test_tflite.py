#!/usr/bin/env python
"""Wrapper fino de CLI - chama handwash.application.check_tflite_viability (GATE, Secao
7.1 do plano: confirma viabilidade de conversao TFLite builtin-only ANTES do treino real).

Nome sem numero (nao "05_...") de proposito - README.md documenta este script como
`scripts/smoke_test_tflite.py`, rodando entre 04_build_windows.py e 05_train_classifier.py
sem tomar o numero 05 (revisao de codigo: a primeira versao numerada colidia com o
"Arquivos criticos" do plano E com o README, que reservam 05 pro treino)."""
import json

from handwash.application.check_tflite_viability import check_builtin_ops_viability

if __name__ == "__main__":
    summary = check_builtin_ops_viability()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["gate_passed"]:
        print(f"\nGATE: PASSOU - {summary['winning_architecture']} converte builtin-only, pode seguir pro treino real.")
    else:
        print("\nGATE: FALHOU - nenhuma arquitetura testada (incluindo Opcao C) converteu builtin-only.")
