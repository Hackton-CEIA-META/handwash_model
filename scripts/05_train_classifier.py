#!/usr/bin/env python
"""Wrapper fino de CLI - chama handwash.application.train_classifier (Secao 5/6 do
plano: treino da Opcao A vencedora do gate TFLite, class weights, augmentation on-the-fly,
avaliacao geral+por classe+matriz de confusao+por video+calibracao).

Uso: python scripts/05_train_classifier.py [max_epochs]
"""
import json
import sys

from handwash.application.train_classifier import train_classifier

if __name__ == "__main__":
    cli_max_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else None
    kwargs = {"max_epochs": cli_max_epochs} if cli_max_epochs is not None else {}
    summary = train_classifier(**kwargs)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
