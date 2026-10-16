"""
Etap 8: podzial na trening / kalibracja / test / zbior_C.
Etap 9: indeks inflacji i cele w skali log.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from .parametry import (
    KLUCZ_SEZON,
    LIGA_ZBIOR_C,
    MIN_OBSERWACJI_KOMORKA,
    PRZEDZIALY_WIEKU,
    SEZONY_KALIBRACJA,
    SEZONY_TEST,
    SEZONY_TRENING,
)

# ETAP 8  podzial


def etap_8_podziel(sezony: pd.DataFrame) -> pd.DataFrame:
    """Dodaje kolumny "podzial" (podzial czasowy, D-16) i "znany_z_treningu" (D-12)."""
    print("\nETAP 8  podzial na trening / kalibracja / test / zbior_C")

    sezony = sezony.copy()

    # Primeira Liga jest pierwsza, wiec trafia do zbioru C niezaleznie od sezonu
    warunki = [
        sezony["liga"] == LIGA_ZBIOR_C,
        sezony["season"].isin(SEZONY_TRENING),
        sezony["season"].isin(SEZONY_KALIBRACJA),
        sezony["season"].isin(SEZONY_TEST),
    ]
    sezony["podzial"] = np.select(
        warunki, ["zbior_C", "trening", "kalibracja", "test"], default=""
    )

    print()
    for nazwa in ["trening", "kalibracja", "test", "zbior_C"]:
        pod = sezony[sezony["podzial"] == nazwa]
        print(
            f"  {nazwa:12s} {len(pod):6d} par ({100 * len(pod) / len(sezony):5.1f}%)  "
            f"{pod['player_id'].nunique():5d} zawodnikow  "
            f"sezony: {', '.join(sorted(pod['season'].unique()))}"
        )

    assert (sezony["podzial"] != "").all(), "sa pary bez przypisanego zbioru"
    for zbior in ["trening", "kalibracja"]:
        ligi = set(sezony.loc[sezony["podzial"] == zbior, "liga"])
        assert LIGA_ZBIOR_C not in ligi, f"{LIGA_ZBIOR_C} w zbiorze {zbior}"
    assert not sezony.duplicated(subset=KLUCZ_SEZON).any(), (
        "ta sama para w wiecej niz jednym zbiorze"
    )

    # Ten sam zawodnik bywa w treningu i w tescie (inne sezony). Flaga sluzy do
    # raportowania wynikow osobno dla znanych i nowych (D-12); nie jest cecha.
    w_treningu = set(sezony.loc[sezony["podzial"] == "trening", "player_id"])
    sezony["znany_z_treningu"] = sezony["player_id"].isin(w_treningu)

    # w samym treningu flaga = False, inaczej kazdy bylby "znany sam sobie"
    sezony.loc[sezony["podzial"] == "trening", "znany_z_treningu"] = False

    print("\n  zawodnicy znani z treningu:")
    for nazwa in ["kalibracja", "test", "zbior_C"]:
        pod = sezony[sezony["podzial"] == nazwa]
        znani = pod["znany_z_treningu"].sum()
        print(
            f"    {nazwa:11s} {znani:5d} z {len(pod):5d} par "
            f"({100 * znani / len(pod):4.1f}%)"
        )

    return sezony


# ETAP 9  deflacja


def policz_indeks_inflacji(trening: pd.DataFrame) -> dict[str, float]:
    """
    Lancuchowy indeks cen o stalym skladzie wiek x pozycja, tylko na treningu.

    Zwraca {sezon: wspolczynnik}, pierwszy sezon = 1.0.
    """
    dane = trening.copy()
    dane["log_wartosc"] = np.log(dane["market_value_in_eur"])
    dane["komorka_wieku"] = pd.cut(dane["wiek"], PRZEDZIALY_WIEKU, right=False)

    sezony_rosnaco = sorted(dane["season"].unique())
    wspolczynniki = {sezony_rosnaco[0]: 1.0}
    biezacy = 1.0

    for wczesniejszy, pozniejszy in pairwise(sezony_rosnaco):
        grupa_a = dane[dane["season"] == wczesniejszy]
        grupa_b = dane[dane["season"] == pozniejszy]

        roznice, wagi = [], []
        for komorka in dane["komorka_wieku"].cat.categories:
            for pozycja in ["DEF", "MID", "FOR"]:
                a = grupa_a[
                    (grupa_a["komorka_wieku"] == komorka)
                    & (grupa_a["pozycja"] == pozycja)
                ]["log_wartosc"]
                b = grupa_b[
                    (grupa_b["komorka_wieku"] == komorka)
                    & (grupa_b["pozycja"] == pozycja)
                ]["log_wartosc"]
                if (
                    len(a) >= MIN_OBSERWACJI_KOMORKA
                    and len(b) >= MIN_OBSERWACJI_KOMORKA
                ):
                    # srednia, nie mediana: TM wycenia w okraglych progach i
                    # mediana skacze miedzy nimi
                    roznice.append(b.mean() - a.mean())
                    wagi.append(len(a))

        zmiana = np.average(roznice, weights=wagi) if roznice else 0.0
        biezacy *= np.exp(zmiana)
        wspolczynniki[pozniejszy] = biezacy
        print(
            f"    {wczesniejszy} -> {pozniejszy}  {100 * (np.exp(zmiana) - 1):>+7.1f}%  "
            f"({len(roznice)} komorek)"
        )

    return wspolczynniki


def etap_9_deflacja(sezony: pd.DataFrame) -> pd.DataFrame:
    """Dodaje market_value_real, log_market_value i log_market_value_real (D-04)."""
    print("\nETAP 9  deflacja (indeks liczony tylko na treningu)")

    sezony = sezony.copy()

    trening = sezony[sezony["podzial"] == "trening"]
    print(f"\n  indeks liczony na {len(trening)} parach treningowych")
    wspolczynniki = policz_indeks_inflacji(trening)

    # sezony spoza treningu dostaja wspolczynnik ostatniego sezonu treningowego,
    # inaczej uzylbym wycen ze zbioru testowego
    ostatni = wspolczynniki[max(wspolczynniki)]
    for sezon in sezony["season"].unique():
        wspolczynniki.setdefault(sezon, ostatni)

    print("\n  wspolczynniki:")
    sezony_treningowe = set(trening["season"])
    for sezon in sorted(wspolczynniki):
        zrodlo = "trening" if sezon in sezony_treningowe else "przedluzenie plaskie"
        print(f"    {sezon}  {wspolczynniki[sezon]:.4f}  ({zrodlo})")

    sezony["market_value_real"] = sezony["market_value_in_eur"] / sezony["season"].map(
        wspolczynniki
    )
    sezony["log_market_value"] = np.log1p(sezony["market_value_in_eur"])
    sezony["log_market_value_real"] = np.log1p(sezony["market_value_real"])

    assert sezony["market_value_real"].gt(0).all(), "niedodatnia wartosc realna"
    assert sezony["log_market_value"].notna().all(), "brak w celu glownym"

    print("\n  cel glowny: log_market_value (nominalny)")
    print(
        f"  skosnosc: surowo {sezony['market_value_in_eur'].skew():.2f}, "
        f"po log {sezony['log_market_value'].skew():.2f}"
    )
    return sezony
