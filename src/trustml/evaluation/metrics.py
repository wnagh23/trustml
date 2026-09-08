"""
Metryki jakosci regresji - wymiar W1.

Zmienna celu zyje w skali log1p(euro) i model uczy sie wlasnie jej. Wszystkie
metryki licza sie na predykcjach surowych, czyli po zwyklym `expm1` tam, gdzie
wynik ma byc w euro. Powod: RMSLE, MdAPE i MAE osiagaja minimum przy medianie
warunkowej, wiec podstawienie pod nie wartosci oczekiwanej pogarsza je z definicji
(D-24, zmierzone na wszystkich poziomach odniesienia).

Korekta Duana zostaje przy kwotach, ktore podaje sie czytelnikowi: wycena
pojedynczego zawodnika w euro i suma wycen skladu. Mechanizm: exp(sredniej
z logarytmow) daje srednia geometryczna, ktora na rozkladzie skosnym lezy ponizej
sredniej arytmetycznej, wiec suma predykcji po samym `expm1` bywa razaco za niska
- dla mediany w komorce wychodzilo 46 procent sumy prawdziwych wycen (D-21).

Zeby dalo sie sprawdzic, czy korekta pomaga akurat temu modelowi, kazdy pomiar
raportuje dwie liczby: `agregat` i `agregat_po_korekcie`, czyli sume predykcji
podzielona przez sume prawdy, bez korekty i z nia. Wartosc bliska 1,0 oznacza
model nieobciazony na poziomie zagregowanym. Dla poziomu M0c korekta psula ten
stosunek z 0,98 na 1,17, bo M0c przepisuje prawdziwa cene i ma juz wlasciwy
rozrzut - stad wymog mierzenia zamiast stosowania w ciemno.

Kolumna `krotnosc` to exp(RMSLE), czyli blad logarytmiczny przelozony na czynnik
mnozacy: 1,70 znaczy "typowo mylimy sie 1,7 raza w gore albo w dol".

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


def mape(y_log, pred_log) -> float:
    """
    Sredni bezwzgledny blad wzgledny w procentach. Miara wadliwa, patrz D-15.

    Zanizenie ma sufit stu procent, zawyzenie sufitu nie ma, wiec model
    optymalizowany pod te miare uczy sie systematycznie zanizac. Raportujemy ja
    wylacznie do porownania z wczesniejszymi podejsciami.
    """
    y, p = _przygotuj(y_log, pred_log)
    prawda = np.expm1(y)
    return float(100.0 * np.mean(np.abs(do_euro(p) - prawda) / prawda))


def mae_euro(y_log, pred_log) -> float:
    """
    Sredni bezwzgledny blad w euro.

    Bez korekty Duana: srednia z wartosci bezwzglednych osiaga minimum przy
    medianie warunkowej, wiec podstawienie wartosci oczekiwanej pogarsza te miare
    z definicji. Zmierzone na wszystkich czterech poziomach odniesienia (D-24).
    """
    y, p = _przygotuj(y_log, pred_log)
    return float(np.mean(np.abs(do_euro(p) - np.expm1(y))))


def krotnosc_bledu(y_log, pred_log) -> float:
    """
    Przeklada RMSLE na czynnik mnozacy, zeby dalo sie go czytac bez logarytmow.

    Wynik 1,70 znaczy "typowo mylimy sie 1,7 raza", czyli predykcja lezy mniej
    wiecej miedzy 59 a 170 procent prawdy. Na euro tego przelozyc sie nie da,
    bo blad logarytmiczny jest bledem wzglednym - te same 0,53 to +-1,6 mln
    u rezerwowego i +-35 mln u gwiazdy.
    """
    return float(np.exp(rmsle(y_log, pred_log)))


def skala_agregatu(y_log, pred_log, duan: float = 1.0) -> float:
    """
    Suma predykcji podzielona przez sume wartosci rzeczywistych, w euro.

    Mierzy obciazenie na poziomie zagregowanym: 0,46 oznacza, ze wyceniajac tym
    modelem sklad, podalibysmy niecala polowe jego prawdziwej wartosci. Jest to
    jedyne miejsce, w ktorym korekta Duana ma sens, i jednoczesnie kryterium
    tego, czy dla danego modelu w ogole jest potrzebna (D-24).

    Przyjmuje:
        y_log    - cel w skali log1p
        pred_log - predykcja w tej samej skali
        duan     - wspolczynnik korekty; 1.0 oznacza pomiar bez niej

    Zwraca:
        stosunek sum, gdzie 1,0 oznacza brak obciazenia.
    """
    y, p = _przygotuj(y_log, pred_log)
    return float(do_euro(p, duan).sum() / np.expm1(y).sum())


def metryki(y_log, pred_log, duan: float = 1.0) -> dict[str, float]:
    """
    Liczy komplet metryk dla jednego zestawu predykcji.

    Przyjmuje:
        y_log    - cel w skali log1p
        pred_log - predykcja w tej samej skali
        duan     - wspolczynnik Duana; wchodzi wylacznie do agregat_po_korekcie

    Zwraca:
        slownik o kluczach z KOLUMNY_METRYK.
    """
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
    Liczy metryki dla calosci i osobno w kazdej grupie na kazdej osi.

    Parametr `grupy` jest obowiazkowy (D-12). Pusty slownik jest dopuszczalny,
    ale musi zostac wpisany swiadomie - chodzi o to, zeby stratyfikacja nie
    ginela przez zapomnienie.

    Przyjmuje:
        y_log    - cel w skali log1p
        pred_log - predykcja w tej samej skali
        grupy    - {nazwa osi: wartosci dla kazdego wiersza}, na przyklad
                   {"znany_z_treningu": ..., "pozycja": ...}
        duan     - wspolczynnik Duana, liczony na resztach treningowych; wchodzi
                   wylacznie do kolumny agregat_po_korekcie
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
