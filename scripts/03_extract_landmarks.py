#!/usr/bin/env python
"""Wrapper fino de CLI - chama handwash.application.extract_landmarks (camada silver, 300 videos)."""
import sys

from handwash.application.extract_landmarks import extract_all

if __name__ == "__main__":
    cli_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    summary = extract_all(limit=cli_limit)
    print(summary)
