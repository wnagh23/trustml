"""
Podzial zbiorow i deflacja zmiennej celu.

ETAP 8  podzial na trening / kalibracja / test / zbior_C
ETAP 9  indeks inflacji i cele w skali logarytmicznej
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
    """
    Przypisuje kazdej parze zawodnik-sezon jeden z czterech zbiorow.

    Podzial czasowy - losowy rozrzucilby tego samego zawodnika po treningu
    i tescie, a jego wycena jest z roku na rok mocno skorelowana.

    Przyjmuje:
        sezony - tabela z etapu 7

    Zwraca:
        te sama tabele z kolumnami "podzial" i "znany_z_treningu".
    """
    print("\nETAP 8  podzial na trening / kalibracja / test / zbior_C")

    sezony = sezony.copy()

    # Kolejnosc warunkow ma znaczenie: Primeira Liga jest pierwsza, wiec wygrywa
    # niezaleznie od sezonu.
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

    # Ten sam zawodnik moze byc w treningu i w innym zbiorze, w roznych sezonach.
    # Pary sa rozlaczne, wiec formalnie podzial jest czysty, ale model rozpoznaje
    # zawodnika po kombinacji wzrostu, wieku, pozycji i profilu statystycznego -
    # mimo ze player_id zostaje poza cechami. Zmierzone HistGradientBoostingiem:
    #   test:    znani R2 0,68 | nowi R2 0,56
    #   zbior_C: znani R2 0,51 | nowi R2 -0,50
    # Przewaga zostaje po wyrownaniu wieku grup, wiec bierze sie z samej
    # znajomosci zawodnika.
    #
    # Podzial zostaje taki, jaki jest - w praktyce wycenia sie zawodnikow,
    # ktorych rynek juz zna. Ale przy raportowaniu: R2 na calym tescie jest
    # wyzsze niz dla zawodnika nowego, a test ma 66% znanych wobec 9% w zbiorze
    # C, wiec uczciwe porownanie dla W2 to nowi z nowymi.
    #
    # Zapisujemy to jako kolumne w zbiorze. Inaczej kazdy notatnik i kazdy modul
    # ewaluacji odtwarzalby zbior player_id z treningu u siebie - wystarczy, ze
    # raz zrobi to po nazwisku zamiast po id, i stratyfikacja przestaje byc
    # porownywalna miedzy raportami.
    # To metadana do analizy, trzymana poza zestawem cech modelu.
    w_treningu = set(sezony.loc[sezony["podzial"] == "trening", "player_id"])
    sezony["znany_z_treningu"] = sezony["player_id"].isin(w_treningu)

    # Wiersze treningowe oznaczamy jako False. Kazdy zawodnik z treningu bylby
    # inaczej "znany sam sobie", a kolumna ma odpowiadac na pytanie: czy model
    # widzial juz tego zawodnika, ZANIM dostal ten konkretny wiersz.
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
    Lancuchowy indeks cen o stalym skladzie wiekowym.

    Dla kazdej pary kolejnych sezonow porownuje srednia log(wartosc) w komorkach
    wiek x pozycja i usrednia roznice wazone liczebnoscia. Porownujemy zawodnikow
    w tym samym wieku, dobieranych osobno w kazdym sezonie - inaczej starzenie sie
    mieszaloby sie z inflacja.

    Przyjmuje:
        trening - wylacznie wiersze treningowe; policzenie indeksu na wszystkich
                  sezonach byloby wyciekiem ze zbioru testowego

    Zwraca:
        slownik {sezon: wspolczynnik}, pierwszy sezon ma 1.0.
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
                    # Srednia: TM wycenia w okraglych progach (123 rozne wartosci
                    # w calym zbiorze), wiec mediana komorki o 60-140 obserwacjach
                    # skacze miedzy progami i daje "inflacje" +60% tam, gdzie
                    # rynek ledwie drgnal.
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
    """
    Dokleja cel zdeflowany i logarytmy obu wersji celu.

    Celem glownym jest log_market_value (nominalny), wersja realna zostaje jako
    kolumna dodatkowa - powody w komentarzu przy jej wyliczeniu.

    Przyjmuje:
        sezony - tabela z etapu 8, z kolumna "podzial"

    Zwraca:
        tabele z market_value_real, log_market_value, log_market_value_real.
    """
    print("\nETAP 9  deflacja (indeks liczony tylko na treningu)")

    sezony = sezony.copy()

    # Funkcja dostaje wylacznie trening - kalibracja, test i zbior C zostaja poza
    # jej zasiegiem.
    trening = sezony[sezony["podzial"] == "trening"]
    print(f"\n  indeks liczony na {len(trening)} parach treningowych")
    wspolczynniki = policz_indeks_inflacji(trening)

    # Sezony spoza treningu dostaja wspolczynnik ostatniego sezonu treningowego.
    # Policzenie ich wlasnych wymagaloby zajrzenia do wycen ze zbioru testowego.
    ostatni = wspolczynniki[max(wspolczynniki)]
    for sezon in sezony["season"].unique():
        wspolczynniki.setdefault(sezon, ostatni)

    print("\n  wspolczynniki:")
    sezony_treningowe = set(trening["season"])
    for sezon in sorted(wspolczynniki):
        zrodlo = "trening" if sezon in sezony_treningowe else "przedluzenie plaskie"
        print(f"    {sezon}  {wspolczynniki[sezon]:.4f}  ({zrodlo})")

    # Inflacja w Big 5 siega 38% miedzy 2017-2018 a 2023-2024, a poziom testu
    # lezy 18,4% powyzej treningu. Deflacja mimo to pogarsza wynik - test na
    # HistGradientBoosting dal MAE 0,5757 wobec 0,5734 na celu nominalnym. Drzewa
    # trzymaja sie zakresu treningu, wiec roznica poziomow przeklada sie
    # na obciazenie slabiej, niz sugeruje porownanie srednich.
    sezony["market_value_real"] = sezony["market_value_in_eur"] / sezony["season"].map(
        wspolczynniki
    )
    # log1p zamiast log - odporny na zero, gdyby kiedys przeszlo przez filtry.
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
