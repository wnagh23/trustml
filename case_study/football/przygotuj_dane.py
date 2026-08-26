"""
Przygotowanie zbioru do modelowania predykcyjnego.

Zrodla:
    1. FBref (statystyki zawodnikow mecz po meczu, master.db)
    2. Transfermarkt (wycena zawodnika z okolic 30 czerwca)

Wyjscie: jeden wiersz na pare (zawodnik, sezon) z cechami opisujacymi jego gre
w tym sezonie oraz wartoscia rynkowa jako zmienna celu.

Skrypt nie modeluje - konczy sie na zapisaniu plikow do data/processed/.

    python case_study/football/przygotuj_dane.py

Etapy sa rozdzielone na moduly:
    zrodla.py    ETAP 1-2   wczytanie FBref, scalenie tabel
    cechy.py     ETAP 3-5   parsowanie, agregacja do sezonu, per 90 minut
    wycena.py    ETAP 6-7   crosswalk do TM, dolaczenie celu
    zbiory.py    ETAP 8-9   podzial zbiorow, indeks inflacji
    ten plik     ETAP 10    zapis wynikow
"""

from __future__ import annotations

import pandas as pd

from cechy import etap_3_parsuj, etap_4_agreguj, etap_5_normalizuj
from parametry import (
    KATALOG_WYJSCIA,
    KLUCZ_SEZON,
    LIGA_ZBIOR_C,
    MIN_MINUT,
    SCIEZKA_DB,
    WSKAZNIKI_PROCENTOWE,
)
from wycena import etap_6_crosswalk, etap_7_dolacz_wycene
from zbiory import etap_8_podziel, etap_9_deflacja
from zrodla import etap_1_wczytaj, etap_2_polacz

# ETAP 10  zapis


def zbuduj_zbior_modelowy(sezony: pd.DataFrame) -> pd.DataFrame:
    """
    Wybiera z pelnej tabeli kolumny wchodzace do modelu.

    Statystyki licznikowe wchodza wylacznie w wersji _p90 - surowe sumy mieszaja
    jakosc zawodnika z liczba rozegranych minut i zostaja w pliku kontrolnym.

    Przyjmuje:
        sezony - tabela z etapu 9

    Zwraca:
        waski DataFrame gotowy do modelowania.
    """
    cechy_p90 = sorted(c for c in sezony.columns if c.endswith("_p90"))
    cechy_procentowe = sorted(WSKAZNIKI_PROCENTOWE)

    kolumny = [
        *KLUCZ_SEZON,
        # metadane do analizy bledow, nie cechy
        "name", "tm_player_id",
        "podzial",
        # cele
        "market_value_in_eur", "market_value_real",
        "log_market_value", "log_market_value_real",
        # cechy liczbowe
        *cechy_p90,
        *cechy_procentowe,
        "wiek", "wiek_do_kw", "wzrost_cm", "minuty_sezon", "mecze",
        # Kategoryczne zostaja tekstem - kodowanie ma sie dziac w Pipeline,
        # zeby kategorie nie rozjechaly sie miedzy zbiorami.
        "pozycja", "region", "liga", "noga", "zmienil_lige",
    ]

    brakujace = [c for c in kolumny if c not in sezony.columns]
    assert not brakujace, f"brak kolumn w zbiorze: {brakujace}"
    return sezony[kolumny]


def etap_10_zapisz(sezony: pd.DataFrame) -> pd.DataFrame:
    """
    Zapisuje pelny zbior kontrolny i waski zbior modelowy.

    Parquet, nie CSV: przy ponad dwustu kolumnach CSV gubi typy, a braki staja sie
    pustymi ciagami nieodroznialnymi od zera.

    Przyjmuje:
        sezony - tabela z etapu 9

    Zwraca:
        zbior modelowy.
    """
    print("\nETAP 10  zapis plikow")

    KATALOG_WYJSCIA.mkdir(parents=True, exist_ok=True)
    sezony = sezony.copy()

    # Wartosc rosnie mniej wiecej do 25 roku zycia i potem spada, wiec zaleznosc
    # nie jest liniowa. Drzewa poradza sobie bez tego, ale modele liniowe nie -
    # dzieki tej cesze porownanie rodzin modeli jest uczciwsze.
    sezony["wiek_do_kw"] = sezony["wiek"] ** 2

    sciezka_pelny = KATALOG_WYJSCIA / "zawodnik_sezon.parquet"
    sezony.to_parquet(sciezka_pelny, index=False)
    print(f"\n  zawodnik_sezon.parquet  {len(sezony):6d} x {sezony.shape[1]:3d}  "
          f"{sciezka_pelny.stat().st_size / 1024 / 1024:.1f} MB")

    model = zbuduj_zbior_modelowy(sezony)
    sciezka_model = KATALOG_WYJSCIA / "model.parquet"
    model.to_parquet(sciezka_model, index=False)
    print(f"  model.parquet           {len(model):6d} x {model.shape[1]:3d}  "
          f"{sciezka_model.stat().st_size / 1024 / 1024:.1f} MB")

    assert not model.duplicated(subset=KLUCZ_SEZON).any(), "klucz nie jest unikalny"
    assert model["log_market_value"].notna().all(), "brak w celu glownym"
    assert (model["minuty_sezon"] >= MIN_MINUT).all(), "para ponizej progu minut"
    for zbior in ["trening", "kalibracja"]:
        assert LIGA_ZBIOR_C not in set(model.loc[model["podzial"] == zbior, "liga"]), (
            f"{LIGA_ZBIOR_C} w zbiorze {zbior}"
        )
    for kolumna in WSKAZNIKI_PROCENTOWE:
        assert model[kolumna].dropna().between(0, 100).all(), f"{kolumna} poza [0,100]"

    print("\n  zbior modelowy:")
    for nazwa, ile in model["podzial"].value_counts().items():
        print(f"    {nazwa:12s} {ile:6d} ({100 * ile / len(model):5.1f}%)")

    braki = model.isna().sum()
    braki = braki[braki > 0].sort_values(ascending=False)
    if len(braki):
        print("\n  braki w cechach (strukturalne, zostaja jako NaN):")
        for kolumna, ile in braki.items():
            print(f"    {kolumna:42s} {ile:5d} ({100 * ile / len(model):4.1f}%)")

    return model


def main() -> pd.DataFrame:
    """Uruchamia caly pipeline i zwraca zbior modelowy."""
    tabele = etap_1_wczytaj(SCIEZKA_DB)
    wystepy = etap_2_polacz(tabele)

    wystepy = etap_3_parsuj(wystepy)
    sezony = etap_4_agreguj(wystepy)
    sezony = etap_5_normalizuj(sezony)

    sezony = etap_6_crosswalk(sezony)
    sezony = etap_7_dolacz_wycene(sezony)

    sezony = etap_8_podziel(sezony)
    sezony = etap_9_deflacja(sezony)

    return etap_10_zapisz(sezony)


if __name__ == "__main__":
    main()
