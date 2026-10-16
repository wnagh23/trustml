"""
Etap 1: wczytanie i deduplikacja tabel z master.db.
Etap 2: polaczenie tabel po (match_id, player_id).
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from .parametry import (
    KLUCZ,
    KOLUMNY_Z_MATCH,
    KOLUMNY_ZREDUNDOWANE,
    TABELE_POLOWE,
    ZMIANY_NAZW,
)

# ETAP 1  wczytanie i deduplikacja


def wczytaj_tabele(polaczenie: sqlite3.Connection, nazwa: str) -> pd.DataFrame:
    """Wczytuje tabele z master.db, usuwa duplikaty i sprawdza unikalnosc klucza."""
    tabela = pd.read_sql(f'SELECT * FROM "{nazwa}"', polaczenie)

    przed = len(tabela)
    tabela = tabela.drop_duplicates()
    usuniete = przed - len(tabela)
    if usuniete:
        print(f"  {nazwa:20s} usunieto {usuniete} zdublowanych wierszy")

    # Match ma tylko match_id, pozostale tabele pelny klucz
    klucz = [k for k in KLUCZ if k in tabela.columns]

    # duplikat klucza o roznej tresci przetrwalby drop_duplicates
    nadmiarowe = tabela.duplicated(subset=klucz).sum()
    assert nadmiarowe == 0, (
        f"{nazwa}: po deduplikacji nadal {nadmiarowe} zdublowanych kluczy {klucz}"
    )
    return tabela


def usun_redundancje(tabele: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Usuwa kolumny powielone miedzy tabelami i zmienia kolidujace nazwy."""
    for nazwa, do_usuniecia in KOLUMNY_ZREDUNDOWANE.items():
        tabele[nazwa] = tabele[nazwa].drop(columns=do_usuniecia)
        print(f"  {nazwa:20s} usunieto {len(do_usuniecia)} zredundowanych kolumn")

    for nazwa, mapowanie in ZMIANY_NAZW.items():
        tabele[nazwa] = tabele[nazwa].rename(columns=mapowanie)
        for stara, nowa in mapowanie.items():
            print(f"  {nazwa:20s} zmiana nazwy: {stara} -> {nowa}")

    gdzie_wystepuje: dict[str, list[str]] = {}
    for nazwa in TABELE_POLOWE:
        for kolumna in tabele[nazwa].columns:
            if kolumna not in KLUCZ:
                gdzie_wystepuje.setdefault(kolumna, []).append(nazwa)

    kolizje = {k: v for k, v in gdzie_wystepuje.items() if len(v) > 1}
    assert not kolizje, f"pozostaly kolizje nazw kolumn: {kolizje}"
    print(f"  kazda z {len(gdzie_wystepuje)} nazw kolumn wystepuje raz")

    return tabele


def etap_1_wczytaj(sciezka_db: Path) -> dict[str, pd.DataFrame]:
    """Wczytuje wszystkie potrzebne tabele z master.db; zwraca {nazwa: DataFrame}."""
    print("\nETAP 1  wczytanie i deduplikacja tabel z master.db")

    assert sciezka_db.exists(), f"nie znaleziono bazy: {sciezka_db}"

    # mode=ro - baze tylko czytam
    uri = f"file:{sciezka_db.as_posix()}?mode=ro"
    tabele: dict[str, pd.DataFrame] = {}

    with closing(sqlite3.connect(uri, uri=True)) as polaczenie:
        print("\n  wczytywanie i deduplikacja wierszy")
        for nazwa in [*TABELE_POLOWE, "Player_Info", "Match"]:
            tabele[nazwa] = wczytaj_tabele(polaczenie, nazwa)

        print("\n  usuwanie kolumn powielonych miedzy tabelami")
        tabele = usun_redundancje(tabele)

    print("\n  stan po etapie:")
    print(f"    {'tabela':22s} {'wierszy':>10s} {'kolumn':>8s}")
    for nazwa, tabela in tabele.items():
        print(f"    {nazwa:22s} {len(tabela):10d} {tabela.shape[1]:8d}")

    assert tabele["Match"]["match_id"].is_unique, "match_id nie jest unikalny w Match"

    liczby = {n: len(tabele[n]) for n in [*TABELE_POLOWE, "Player_Info"]}
    assert len(set(liczby.values())) == 1, f"rozne liczby wierszy: {liczby}"

    return tabele


# ETAP 2  polaczenie tabel


def etap_2_polacz(tabele: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Laczy tabele wystepow i dokleja dane meczu; wiersz = wystep zawodnika w meczu."""
    print("\nETAP 2  polaczenie tabel po (match_id, player_id)")

    polaczone = tabele["Player_Info"]
    oczekiwane = len(polaczone)
    print(f"\n  start: Player_Info, {oczekiwane} wystepow, {polaczone.shape[1]} kolumn")

    for nazwa in TABELE_POLOWE:
        przed = polaczone.shape[1]
        polaczone = polaczone.merge(
            tabele[nazwa], on=KLUCZ, how="inner", validate="one_to_one"
        )
        assert len(polaczone) == oczekiwane, (
            f"po dolaczeniu {nazwa} jest {len(polaczone)} wierszy zamiast "
            f"{oczekiwane} - zbiory kluczy przestaly byc identyczne"
        )
        print(
            f"  + {nazwa:20s} {polaczone.shape[1] - przed:3d} nowych kolumn "
            f"-> {polaczone.shape[1]:3d} kolumn"
        )

    przed = polaczone.shape[1]
    polaczone = polaczone.merge(
        tabele["Match"][KOLUMNY_Z_MATCH],
        on="match_id",
        how="inner",
        validate="many_to_one",
    )
    assert len(polaczone) == oczekiwane, (
        f"po dolaczeniu Match jest {len(polaczone)} wierszy zamiast {oczekiwane}"
    )
    print(
        f"  + {'Match':20s} {polaczone.shape[1] - przed:3d} nowych kolumn "
        f"-> {polaczone.shape[1]:3d} kolumn"
    )

    # sufiksy _x/_y oznaczalyby niepelna liste redundancji
    sufiksy = [c for c in polaczone.columns if c.endswith(("_x", "_y"))]
    assert not sufiksy, f"laczenie wyprodukowalo zdublowane kolumny: {sufiksy}"
    assert not polaczone.duplicated(subset=KLUCZ).any(), "klucz nie jest unikalny"

    print("\n  stan po etapie:")
    print(f"    wystepow:    {len(polaczone)}")
    print(f"    kolumn:      {polaczone.shape[1]}")
    print(f"    zawodnikow:  {polaczone['player_id'].nunique()}")
    print(f"    meczow:      {polaczone['match_id'].nunique()}")
    print(f"    sezonow:     {polaczone['season'].nunique()}")

    return polaczone
