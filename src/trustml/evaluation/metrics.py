"""
Metryki jakosci regresji - wymiar W1.

Zmienna celu zyje w skali log1p(euro) i model uczy sie wlasnie jej. Czesc miar
ma sens w tej skali, czesc dopiero po powrocie do euro, a powrot nie jest zwyklym
odwroceniem funkcji: exp(sredniej z logarytmow) daje srednia geometryczna, ktora
na rozkladzie skosnym lezy ponizej sredniej arytmetycznej. Korekta Duana mnozy
wynik przez srednia z exp(reszt treningowych) i przywraca nieobciazona wartosc
oczekiwana w euro (D-21).

Podzial obowiazujacy w calej pracy:

    rmsle, r2_log     skala logarytmiczna, predykcje surowe
    mdape             euro, predykcje surowe - miara medianowa
    mae_eur, mape     euro, predykcje po korekcie Duana

Miary medianowe korekty nie dostaja. Duan przesuwa predykcje w gore, zeby dawaly
poprawna srednia; mediana warunkowa lezy nizej, wiec doklejenie korekty przed
policzeniem MdAPE systematycznie pogorszyloby wynik bez zysku interpretacyjnego.

Funkcja `ocen` wymaga podania osi stratyfikacji (D-12). Jedna liczba dla calego
zbioru chowa fakt, ze model zachowuje sie inaczej na zawodnikach znanych
z treningu niz na nowych - na zbiorze C roznica siegala R2 0,51 wobec -0,50.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

# Kolejnosc kolumn w raportach. RMSLE i MdAPE sa glowne (D-15), R2 i MAE
# pomocnicze, MAPE wylacznie do porownan z wczesniejszymi podejsciami - dzieli
# przez wartosc rzeczywista, wiec na skosnym rozkladzie mierzy glownie zachowanie
# modelu na tanich zawodnikach.
KOLUMNY_METRYK = ("n", "rmsle", "mdape", "r2_log", "mae_eur", "mape")


def _przygotuj(y_log, pred_log) -> tuple[np.ndarray, np.ndarray]:
    """
    Sprowadza oba wejscia do plaskich tablic float i sprawdza ich kompletnosc.

    Brak w predykcji jest bledem, a nie sytuacja do cichego pominiecia: baseline
    M0c nie ma czego przepisac dla zawodnika bez poprzedniego sezonu i decyzja,
    co z takim wierszem zrobic, nalezy do wolajacego (D-06).
    """
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
    """
    Liczy wspolczynnik korekty Duana ze srednich reszt w skali logarytmicznej.

    Reszty musza pochodzic ze zbioru treningowego. Policzenie ich na tescie
    wymagaloby etykiet testowych, czyli byloby wyciekiem.

    Przyjmuje:
        reszty_log - y_log minus predykcja, oba w skali log1p

    Zwraca:
        skalar, typowo tuz powyzej 1,0.
    """
    r = np.asarray(reszty_log, dtype=float).ravel()
    if r.size == 0:
        raise ValueError("puste reszty")
    if not np.isfinite(r).all():
        raise ValueError("brak albo nieskonczonosc w resztach")
    return float(np.mean(np.exp(r)))


def do_euro(pred_log, duan: float = 1.0) -> np.ndarray:
    """
    Wraca ze skali log1p do euro, opcjonalnie z korekta Duana.

    Wzor bierze sie z rozlozenia celu na predykcje i reszte: log1p(y) = pred + e,
    czyli y = exp(pred) * exp(e) - 1. Odruchowe `expm1(pred) * duan` przemnozyloby
    rowniez te jedynke i jest formalnie zle.

    Przyjmuje:
        pred_log - predykcja w skali log1p
        duan     - wspolczynnik z `wspolczynnik_duana`; 1.0 oznacza brak korekty

    Zwraca:
        tablice wartosci w euro.
    """
    return np.exp(np.asarray(pred_log, dtype=float).ravel()) * float(duan) - 1.0


def rmsle(y_log, pred_log) -> float:
    """
    Pierwiastek ze sredniego kwadratu bledu w skali logarytmicznej.

    Cel jest juz zapisany jako log1p(euro), wiec RMSLE sprowadza sie do RMSE
    liczonego w tej samej skali, w ktorej pracuje model.
    """
    y, p = _przygotuj(y_log, pred_log)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def r2_log(y_log, pred_log) -> float:
    """Udzial wariancji celu wyjasniony przez predykcje, w skali logarytmicznej."""
    y, p = _przygotuj(y_log, pred_log)
    calkowita = float(np.sum((y - y.mean()) ** 2))
    if calkowita == 0.0:
        return float("nan")
    return float(1.0 - np.sum((y - p) ** 2) / calkowita)


def mdape(y_log, pred_log) -> float:
    """
    Mediana bezwzglednego bledu wzglednego w procentach, bez korekty Duana.

    Czyta sie po ludzku: polowa wycen miesci sie w tylu procentach od prawdy.
    """
    y, p = _przygotuj(y_log, pred_log)
    prawda = np.expm1(y)
    if not (prawda > 0).all():
        raise ValueError("niedodatnia wartosc rzeczywista - blad wzgledny bez sensu")
    return float(100.0 * np.median(np.abs(do_euro(p) - prawda) / prawda))


def mape(y_log, pred_log, duan: float = 1.0) -> float:
    """
    Sredni bezwzgledny blad wzgledny w procentach. Miara wadliwa, patrz D-15.

    Zanizenie ma sufit stu procent, zawyzenie sufitu nie ma, wiec model
    optymalizowany pod te miare uczy sie systematycznie zanizac. Raportujemy ja
    wylacznie do porownania z wczesniejszymi podejsciami.
    """
    y, p = _przygotuj(y_log, pred_log)
    prawda = np.expm1(y)
    return float(100.0 * np.mean(np.abs(do_euro(p, duan) - prawda) / prawda))


def mae_euro(y_log, pred_log, duan: float = 1.0) -> float:
    """Sredni bezwzgledny blad w euro, po korekcie Duana."""
    y, p = _przygotuj(y_log, pred_log)
    return float(np.mean(np.abs(do_euro(p, duan) - np.expm1(y))))


def metryki(y_log, pred_log, duan: float = 1.0) -> dict[str, float]:
    """
    Liczy komplet metryk dla jednego zestawu predykcji.

    Przyjmuje:
        y_log    - cel w skali log1p
        pred_log - predykcja w tej samej skali
        duan     - wspolczynnik korekty dla miar w euro

    Zwraca:
        slownik o kluczach z KOLUMNY_METRYK.
    """
    y, p = _przygotuj(y_log, pred_log)
    return {
        "n": int(y.size),
        "rmsle": rmsle(y, p),
        "mdape": mdape(y, p),
        "r2_log": r2_log(y, p),
        "mae_eur": mae_euro(y, p, duan),
        "mape": mape(y, p, duan),
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
    Liczy metryki dla calosci i osobno w kazdej grupie na kazdej osi.

    Parametr `grupy` jest obowiazkowy (D-12). Pusty slownik jest dopuszczalny,
    ale musi zostac wpisany swiadomie - chodzi o to, zeby stratyfikacja nie
    ginela przez zapomnienie.

    Przyjmuje:
        y_log    - cel w skali log1p
        pred_log - predykcja w tej samej skali
        grupy    - {nazwa osi: wartosci dla kazdego wiersza}, na przyklad
                   {"znany_z_treningu": ..., "pozycja": ...}
        duan     - wspolczynnik korekty dla miar w euro
        etykieta - nazwa modelu, trafia do kolumny "model"

    Zwraca:
        ramke z kolumnami model, os, grupa, duan i metrykami; pierwszy wiersz
        opisuje zawsze calosc zbioru.
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
