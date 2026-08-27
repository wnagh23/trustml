"""
Metryczka uruchomienia pipeline'u.

Zapisuje wszystko, co pozwala odtworzyc zbior i sprawdzic, czy dwa przebiegi
dotyczyly tych samych danych: odciski plikow wejsciowych, wersje bibliotek,
parametry i liczby wierszy po kazdym etapie.

Dzieki temu pytanie "czemu dzis wyszlo 13 640 par wobec 13 709" ma odpowiedz
w pliku: widac, czy zmienil sie prog, biblioteka, czy plik zrodlowy.
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
    """
    Liczy odcisk SHA-256 pliku.

    Czyta kawalkami, bo master.db ma 211 MB i wczytanie go w calosci do pamieci
    byloby marnotrawstwem.

    Przyjmuje:
        sciezka - plik do zahaszowania
        kawalek - rozmiar porcji w bajtach

    Zwraca:
        odcisk jako 64 znaki szesnastkowe.
    """
    skrot = hashlib.sha256()
    with sciezka.open("rb") as plik:
        while porcja := plik.read(kawalek):
            skrot.update(porcja)
    return skrot.hexdigest()


def opisz_plik(sciezka: Path) -> dict:
    """
    Zwraca odcisk, rozmiar i date modyfikacji pliku.

    Przyjmuje:
        sciezka - plik wejsciowy albo wyjsciowy

    Zwraca:
        slownik z metadanymi; przy braku pliku samo {"brak": True}.
    """
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
    """
    Zbiera wszystkie stale z modulu parametry.

    Bierzemy nazwy pisane wielkimi literami - taka jest konwencja stalych
    w Pythonie, wiec lista aktualizuje sie sama, gdy dojdzie nowy parametr.
    Wartosci nieserializowalne (sciezki, zbiory) zamieniamy na tekst i listy.

    Zwraca:
        slownik {nazwa: wartosc} gotowy do zapisania w JSON.
    """
    zebrane = {}
    for nazwa in dir(par):
        if not nazwa.isupper():
            continue
        wartosc = getattr(par, nazwa)
        if isinstance(wartosc, Path):
            # Sciezki zapisujemy wzglednie - absolutne zdradzalyby uklad dyskow
            # i roznilyby sie miedzy maszynami przy porownywaniu manifestow.
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
    """
    Sklada manifest z metadanych srodowiska, danych i przebiegu.

    Przyjmuje:
        wiersze_po_etapach - licznik wierszy zebrany w trakcie przebiegu
        pliki_wyjsciowe    - sciezki zapisanych plikow

    Zwraca:
        slownik gotowy do zapisania jako JSON.
    """
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

    # Prowieniencja FBref z audytu archiwow, jesli zostal uruchomiony. To ona
    # mowi, skad wziela sie master.db - sama baza przychodzi bez metadanych.
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
    """
    Buduje manifest i zapisuje go obok plikow wynikowych.

    Przyjmuje:
        wiersze_po_etapach - licznik wierszy z przebiegu
        pliki_wyjsciowe    - sciezki zapisanych plikow

    Zwraca:
        sciezke zapisanego manifestu.
    """
    manifest = zbuduj_manifest(wiersze_po_etapach, pliki_wyjsciowe)
    sciezka = par.KATALOG_WYJSCIA / "manifest.json"
    sciezka.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return sciezka
