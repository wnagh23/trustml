"""
Poziomy odniesienia M0a-M0c (D-06). Wszystkie licza na celu w skali log.
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
    """M0a: mediana celu z treningu dla kazdego wiersza z `docelowe`."""
    return np.full(len(docelowe), float(trening[cel].median()))


def mediana_komorki(
    trening: pd.DataFrame,
    docelowe: pd.DataFrame,
    cel: str,
    kolumny: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    M0b: mediana celu w komorce wyznaczonej przez `kolumny`.

    Komorki nieobecne w treningu dostaja mediane globalna.
    Zwraca (predykcje, maska wierszy uzupelnionych).
    """
    kolumny = list(kolumny)
    # bez observed=True groupby tworzy wszystkie kombinacje poziomow, takze puste
    mediany = trening.groupby(kolumny, observed=True)[cel].median()
    # copy=True, bo pandas 3.0 zwraca tu widok tylko do odczytu
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
    M0c: wartosc tej samej jednostki z poprzedniego okresu.

    Poprzedni okres liczy funkcja `poprzedni_okres`, wiec luka 2019-2020 daje NaN.
    Zwraca (predykcje z NaN tam, gdzie brak poprzednika, maska pokrycia).
    """
    if historia.duplicated(subset=[klucz, okres]).any():
        raise ValueError(f"historia ma powtorzone pary ({klucz}, {okres})")

    zrodlo = historia.set_index([klucz, okres])[cel]
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
    Wypelnia braki w `predykcje` wartosciami z `zapasowe` (wariant M0c_pelny).

    Zwraca (predykcje uzupelnione, maska wierszy uzupelnionych).
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
