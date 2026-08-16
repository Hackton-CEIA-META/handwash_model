#!/usr/bin/env python
"""Wrapper fino de CLI - chama handwash.application.build_windows (camada gold: janelas
normalizadas de 2.0s/30 passos a 15fps, split por sujeito ja aplicado)."""
import sys

from handwash.application.build_windows import build_windows

if __name__ == "__main__":
    cli_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    summary = build_windows(limit=cli_limit)
    print(summary)
