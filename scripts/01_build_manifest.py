#!/usr/bin/env python
"""Wrapper fino de CLI - chama handwash.application.build_manifest."""
from collections import Counter

from handwash.application.build_manifest import build_manifest
from handwash.config.paths import BRONZE_MANIFEST

if __name__ == "__main__":
    rows = build_manifest()
    class_counts = Counter(r["merged_class"] for r in rows)
    split_counts = Counter(r["split"] for r in rows)
    print(f"{len(rows)} videos catalogados em {BRONZE_MANIFEST}")
    print(f"Contagem por classe: {dict(class_counts)}")
    print(f"Contagem por split: {dict(split_counts)}")
