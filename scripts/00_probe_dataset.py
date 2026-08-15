#!/usr/bin/env python
"""Wrapper fino de CLI - chama handwash.application.probe_dataset."""
from handwash.application.probe_dataset import run_probe

if __name__ == "__main__":
    summary = run_probe()
    print(f"Videos verificados: {summary['total_videos']}")
    print(f"Distribuicao de fps: {summary['fps_distribution']}")
    print(f"Distribuicao de resolucao: {summary['resolution_distribution']}")
    print(f"Relatorio salvo em: {summary['output_csv']}")
