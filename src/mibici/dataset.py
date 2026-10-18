"""Punto de entrada de la ingesta reproducible de datos de MiBici.

Este módulo conserva el nombre solicitado por la rúbrica y delega la lógica
robusta de lectura y consolidación en ``mibici.ingestion``.
"""

from __future__ import annotations

from mibici.ingestion import load_raw, main, read_csv_robust, save_consolidated

__all__ = ["load_raw", "main", "read_csv_robust", "save_consolidated"]


if __name__ == "__main__":
    main()
