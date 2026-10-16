"""
Manifest przebiegu pipeline'u: SHA-256 plikow wejsciowych i wyjsciowych,
wersje bibliotek, parametry i liczby wierszy po etapach.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import parametry as par


def policz_sha256(sciezka: Path, kawalek: int = 1024 * 1024) -> str:
    """Liczy SHA-256 pliku porcjami (master.db ma ponad 200 MB)."""
    skrot = hashlib.sha256()
    with sciezka.open("rb") as plik:
        while porcja := plik.read(kawalek):
            skrot.update(porcja)
    return skrot.hexdigest()


def opisz_plik(sciezka: Path) -> dict:
    """Zwraca SHA-256, rozmiar i date modyfikacji pliku albo {"brak": True}."""
    if not sciezka.exists():
        return {"brak": True}
    dane = sciezka.stat()
    return {
        "sha256": policz_sha256(sciezka),
        "rozmiar_mb": round(dane.st_size / 1024 / 1024, 2),
        "zmodyfikowany": datetime.fromtimestamp(dane.st_mtime, UTC).isoformat(
            timespec="seconds"
        ),
    }


def zbierz_parametry() -> dict:
    """Zbiera stale (nazwy wielkimi literami) z parametry.py w postaci do JSON."""
    zebrane = {}
    for nazwa in dir(par):
        if not nazwa.isupper():
            continue
        wartosc = getattr(par, nazwa)
        if isinstance(wartosc, Path):
            # sciezki wzgledne, zeby manifesty z roznych komputerow dalo sie porownac
            try:
                wartosc = str(wartosc.relative_to(par.KATALOG_REPO))
            except ValueError:
                wartosc = str(wartosc)
        elif isinstance(wartosc, frozenset | set):
            wartosc = sorted(wartosc)
        elif isinstance(wartosc, dict):
            wartosc = {
                k: list(v) if isinstance(v, tuple) else v for k, v in wartosc.items()
            }
        zebrane[nazwa] = wartosc
    return zebrane


def zbuduj_manifest(wiersze_po_etapach: dict[str, int], pliki_wyjsciowe: dict) -> dict:
    """Sklada slownik manifestu z danych o srodowisku, plikach i przebiegu."""
    wejscie = {
        "master.db": par.SCIEZKA_DB,
        "players.csv": par.SCIEZKA_TM / "players.csv",
        "player_valuations.csv": par.SCIEZKA_TM / "player_valuations.csv",
    }

    manifest = {
        "uruchomiono": datetime.now(UTC).isoformat(timespec="seconds"),
        "srodowisko": {
            "python": sys.version.split()[0],
            "platforma": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "pliki_wejsciowe": {n: opisz_plik(s) for n, s in wejscie.items()},
        "parametry": zbierz_parametry(),
        "wiersze_po_etapach": wiersze_po_etapach,
        "pliki_wyjsciowe": {n: opisz_plik(s) for n, s in pliki_wyjsciowe.items()},
    }

    # wynik scripts/audyt_zrodel.py - skad pochodzi master.db
    audyt = par.KATALOG_REPO / "docs" / "zrodla_fbref.json"
    if audyt.exists():
        dane = json.loads(audyt.read_text(encoding="utf-8"))
        manifest["prowieniencja_fbref"] = {
            "wzor_adresu": dane["wzor_adresu"],
            "scrape_od": dane["scrape_od"],
            "scrape_do": dane["scrape_do"],
            "stron_lacznie": dane["stron_lacznie"],
            "zgodnosc_z_baza": dane["meczow_bez_strony"] == 0
            and dane["stron_bez_meczu_w_bazie"] == 0,
        }

    return manifest


def zapisz_manifest(wiersze_po_etapach: dict[str, int], pliki_wyjsciowe: dict) -> Path:
    """Zapisuje manifest.json w data/processed/ i zwraca jego sciezke."""
    manifest = zbuduj_manifest(wiersze_po_etapach, pliki_wyjsciowe)
    sciezka = par.KATALOG_WYJSCIA / "manifest.json"
    sciezka.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return sciezka
