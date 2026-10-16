"""
Metryki regresji (W1). Cel i predykcje sa w skali log1p(euro).

Metryki licze na predykcjach bez korekty Duana (D-24). Korekta wchodzi tylko
do `agregat_po_korekcie`, czyli do sum wycen w euro.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

KOLUMNY_METRYK = (
    "n",
    "rmsle",
    "krotnosc",
    "mdape",
    "r2_log",
    "mae_eur",
    "mape",
    "agregat",
    "agregat_po_korekcie",
)


def _przygotuj(y_log, pred_log) -> tuple[np.ndarray, np.ndarray]:
    """Zamienia wejscia na plaskie tablice float i sprawdza, czy nie ma brakow."""
    y = np.asarray(y_log, dtype=float).ravel()
    p = np.asarray(pred_log, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"rozne dlugosci: cel {y.shape}, predykcja {p.shape}")
    if y.size == 0:
        raise ValueError("pusty zbior do oceny")
    if not np.isfinite(y).all():
        raise ValueError("brak albo nieskonczonosc w zmiennej celu")
    if not np.isfinite(p).all():
        raise ValueError(
            "brak w predykcji - zawez ocene do podzbioru z pokryciem "
            "albo uzupelnij predykcje jawnie"
        )
    return y, p


def wspolczynnik_duana(reszty_log) -> float:
    """Wspolczynnik Duana: srednia z exp(reszt). Reszty tylko z treningu."""
    r = np.asarray(reszty_log, dtype=float).ravel()
    if r.size == 0:
        raise ValueError("puste reszty")
    if not np.isfinite(r).all():
        raise ValueError("brak albo nieskonczonosc w resztach")
    return float(np.mean(np.exp(r)))


def do_euro(pred_log, duan: float = 1.0) -> np.ndarray:
    """Zamienia predykcje z log1p na euro; `duan` = 1.0 oznacza brak korekty."""
    # log1p(y) = pred + e, wiec y = exp(pred) * exp(e) - 1; mnozymy przed odjeciem 1
    return np.exp(np.asarray(pred_log, dtype=float).ravel()) * float(duan) - 1.0


def rmsle(y_log, pred_log) -> float:
    """RMSE w skali log1p, czyli RMSLE w euro."""
    y, p = _przygotuj(y_log, pred_log)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def r2_log(y_log, pred_log) -> float:
    """R2 w skali logarytmicznej."""
    y, p = _przygotuj(y_log, pred_log)
    calkowita = float(np.sum((y - y.mean()) ** 2))
    if calkowita == 0.0:
        return float("nan")
    return float(1.0 - np.sum((y - p) ** 2) / calkowita)


def mdape(y_log, pred_log) -> float:
    """Mediana bezwzglednego bledu procentowego w euro."""
    y, p = _przygotuj(y_log, pred_log)
    prawda = np.expm1(y)
    if not (prawda > 0).all():
        raise ValueError("niedodatnia wartosc rzeczywista - blad wzgledny bez sensu")
    return float(100.0 * np.median(np.abs(do_euro(p) - prawda) / prawda))


def mape(y_log, pred_log) -> float:
    """Sredni bezwzgledny blad procentowy. Tylko do porownan (D-15)."""
    y, p = _przygotuj(y_log, pred_log)
    prawda = np.expm1(y)
    return float(100.0 * np.mean(np.abs(do_euro(p) - prawda) / prawda))


def mae_euro(y_log, pred_log) -> float:
    """Sredni bezwzgledny blad w euro, bez korekty Duana (D-24)."""
    y, p = _przygotuj(y_log, pred_log)
    return float(np.mean(np.abs(do_euro(p) - np.expm1(y))))


def krotnosc_bledu(y_log, pred_log) -> float:
    """exp(RMSLE): o ile razy typowo mylimy sie w gore albo w dol."""
    return float(np.exp(rmsle(y_log, pred_log)))


def skala_agregatu(y_log, pred_log, duan: float = 1.0) -> float:
    """Suma predykcji / suma prawdziwych wartosci w euro; 1.0 = brak obciazenia."""
    y, p = _przygotuj(y_log, pred_log)
    return float(do_euro(p, duan).sum() / np.expm1(y).sum())


def metryki(y_log, pred_log, duan: float = 1.0) -> dict[str, float]:
    """Liczy wszystkie metryki z KOLUMNY_METRYK dla jednego zestawu predykcji."""
    y, p = _przygotuj(y_log, pred_log)
    return {
        "n": int(y.size),
        "rmsle": rmsle(y, p),
        "krotnosc": krotnosc_bledu(y, p),
        "mdape": mdape(y, p),
        "r2_log": r2_log(y, p),
        "mae_eur": mae_euro(y, p),
        "mape": mape(y, p),
        "agregat": skala_agregatu(y, p),
        "agregat_po_korekcie": skala_agregatu(y, p, duan),
    }


def ocen(
    y_log,
    pred_log,
    grupy: Mapping[str, object],
    *,
    duan: float = 1.0,
    etykieta: str = "",
) -> pd.DataFrame:
    """
    Liczy metryki dla calosci i dla kazdej grupy na kazdej osi z `grupy` (D-12).

    `grupy` to {nazwa osi: wartosci dla wierszy}. Pierwszy wiersz wyniku to calosc.
    """
    y, p = _przygotuj(y_log, pred_log)

    wiersze = [
        {
            "model": etykieta,
            "os": "calosc",
            "grupa": "wszystko",
            "duan": float(duan),
            **metryki(y, p, duan),
        }
    ]

    for nazwa_osi, wartosci in grupy.items():
        seria = pd.Series(np.asarray(wartosci).ravel())
        if len(seria) != y.size:
            raise ValueError(
                f"os {nazwa_osi} ma {len(seria)} wartosci wobec {y.size} wierszy"
            )
        # sortowanie po str, bo os bywa logiczna, tekstowa albo liczbowa
        for wartosc in sorted(pd.unique(seria.dropna()), key=str):
            maska = (seria == wartosc).to_numpy()
            wiersze.append(
                {
                    "model": etykieta,
                    "os": nazwa_osi,
                    "grupa": str(wartosc),
                    "duan": float(duan),
                    **metryki(y[maska], p[maska], duan),
                }
            )

    return pd.DataFrame(wiersze)
