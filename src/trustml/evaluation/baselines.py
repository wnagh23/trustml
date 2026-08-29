"""
Poziomy odniesienia - modele trywialne, do ktorych porownujemy wszystko inne.

Sens tego modulu jest taki: liczba w rodzaju "R2 0,88" nic nie znaczy, dopoki
nie wiadomo, ile daje odpowiedz udzielona bez zadnej wiedzy o grze. Trzy
poziomy odniesienia z planu E4:

    M0a  mediana globalna ze zbioru treningowego
    M0b  mediana w komorce, na przyklad pozycja x liga
    M0c  przepisanie wartosci tej samej jednostki z poprzedniego okresu

Wszystkie licza mediane celu w skali logarytmicznej. Mediana jest ekwiwariantna
wzgledem przeksztalcen monotonicznych, wiec mediana logarytmow to logarytm
mediany - kolejnosc operacji nie ma tu znaczenia.

M0c korzysta z informacji swiadomie odcietej modelom (D-05) i obejmuje tylko te
wiersze, dla ktorych poprzedni okres istnieje w danych. To inne zadanie z dostepem
do innej wiedzy, wiec raportujemy je jako punkt odniesienia i nie traktujemy jak
konkurenta (D-06).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def _klucz_indeksu(ramka: pd.DataFrame, kolumny: Sequence[str]) -> pd.Index:
    """Buduje indeks pasujacy do wyniku groupby po tych samych kolumnach."""
    if len(kolumny) == 1:
        return pd.Index(ramka[kolumny[0]])
    return pd.MultiIndex.from_frame(ramka[list(kolumny)])


def mediana_globalna(trening: pd.DataFrame, docelowe: pd.DataFrame, cel: str):
    """
    M0a - kazdemu wierszowi przypisuje te sama mediane ze zbioru treningowego.

    Przyjmuje:
        trening  - wiersze treningowe
        docelowe - wiersze do wycenienia
        cel      - nazwa kolumny celu w skali logarytmicznej

    Zwraca:
        tablice predykcji o dlugosci `docelowe`.
    """
    return np.full(len(docelowe), float(trening[cel].median()))


def mediana_komorki(
    trening: pd.DataFrame,
    docelowe: pd.DataFrame,
    cel: str,
    kolumny: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    M0b - mediana celu w komorce wyznaczonej przez wskazane kolumny.

    Komorki nieobecne w treningu dostaja mediane globalna. Przy podziale czasowym
    warto pamietac, ze sezon zbioru docelowego z definicji nie wystepuje
    w treningu, wiec nie moze byc jedna z kolumn komorki.

    Przyjmuje:
        trening  - wiersze treningowe
        docelowe - wiersze do wycenienia
        cel      - nazwa kolumny celu w skali logarytmicznej
        kolumny  - kolumny wyznaczajace komorke, na przyklad pozycja i liga

    Zwraca:
        pare (predykcje, maska wierszy uzupelnionych mediana globalna).
    """
    kolumny = list(kolumny)
    # observed=True liczy wylacznie kombinacje faktycznie wystepujace w danych;
    # przy kolumnach kategorycznych domyslne zachowanie tworzy iloczyn kartezjanski
    # wszystkich poziomow i zasypuje wynik pustymi komorkami.
    mediany = trening.groupby(kolumny, observed=True)[cel].median()
    # copy=True, bo pandas 3.0 oddaje widok tylko do odczytu, a ponizej
    # nadpisujemy brakujace komorki.
    predykcje = mediany.reindex(_klucz_indeksu(docelowe, kolumny)).to_numpy(
        dtype=float, na_value=np.nan, copy=True
    )

    brakujace = ~np.isfinite(predykcje)
    predykcje[brakujace] = float(trening[cel].median())
    return predykcje, brakujace


def przepisanie_z_poprzedniego(
    historia: pd.DataFrame,
    docelowe: pd.DataFrame,
    *,
    klucz: str,
    okres: str,
    cel: str,
    poprzedni_okres: Callable[[object], object],
) -> tuple[np.ndarray, np.ndarray]:
    """
    M0c - przepisuje wartosc tej samej jednostki z poprzedniego okresu.

    Poprzedni okres wyznacza funkcja podana przez wolajacego, a nie przesuniecie
    o jedna pozycje na liscie okresow. Roznica ma znaczenie tam, gdzie w danych
    jest luka kalendarzowa: przesuniecie pozycyjne po cichu skleiloby sezony
    odlegle o dwa lata (D-08), a funkcja parsujaca okres zwroci okres nieobecny
    w danych i wiersz uczciwie zostanie bez predykcji.

    Przyjmuje:
        historia        - wiersze, z ktorych wolno czerpac wartosci
        docelowe        - wiersze do wycenienia
        klucz           - kolumna jednostki, zwykle player_id
        okres           - kolumna czasu, zwykle season
        cel             - nazwa kolumny celu w skali logarytmicznej
        poprzedni_okres - funkcja okres -> okres poprzedni

    Zwraca:
        pare (predykcje z NaN tam, gdzie brak poprzednika, maska pokrycia).
    """
    if historia.duplicated(subset=[klucz, okres]).any():
        raise ValueError(f"historia ma powtorzone pary ({klucz}, {okres})")

    zrodlo = historia.set_index([klucz, okres])[cel]
    # map z funkcja stosuje ja do kazdej wartosci serii z osobna
    szukane = pd.MultiIndex.from_arrays(
        [docelowe[klucz], docelowe[okres].map(poprzedni_okres)]
    )
    predykcje = zrodlo.reindex(szukane).to_numpy(
        dtype=float, na_value=np.nan, copy=True
    )
    return predykcje, np.isfinite(predykcje)


def uzupelnij(
    predykcje: np.ndarray,
    zapasowe: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wypelnia braki jednej predykcji wartosciami drugiej.

    Sluzy wariantowi "M0c na pelnym zbiorze z uzupelnieniem M0b" z D-06.
    Podawanie samego wyniku na podzbiorze z pokryciem, bez wersji na calosci,
    jest mylace - baseline wyglada wtedy lepiej, niz jest, bo trudne wiersze
    bez historii po prostu z niego wypadaja.

    Przyjmuje:
        predykcje - tablica z brakami
        zapasowe  - tablica bez brakow, tej samej dlugosci

    Zwraca:
        pare (predykcje uzupelnione, maska wierszy uzupelnionych).
    """
    predykcje = np.asarray(predykcje, dtype=float).copy()
    zapasowe = np.asarray(zapasowe, dtype=float)
    if predykcje.shape != zapasowe.shape:
        raise ValueError("rozne dlugosci predykcji i zrodla uzupelnienia")

    brakujace = ~np.isfinite(predykcje)
    predykcje[brakujace] = zapasowe[brakujace]
    if not np.isfinite(predykcje).all():
        raise ValueError("zrodlo uzupelnienia samo ma braki")
    return predykcje, brakujace
