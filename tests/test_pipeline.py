"""
Testy wspolnego Pipeline przetwarzania cech (etap E5).

    pytest tests/test_pipeline.py -v

Trzy rzeczy sa tu warte pilnowania, bo psuja sie po cichu: kodowanie kategorii
z upuszczonym poziomem (D-10), zgubienie flagi braku (O-3) i niedeterministyczny
filtr korelacyjny, ktory rozchwialby ranking waznosci cech miedzy przebiegami.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from trustml.models import FiltrKorelacyjny, zbuduj_przetwarzanie


@pytest.fixture
def ramka() -> pd.DataFrame:
    """Cztery kolumny liczbowe, dwie kategoryczne, jedna niewymieniona nigdzie."""
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            # b jest dokladna kopia a przemnozona przez dwa
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
            "c": [5.0, 1.0, 4.0, 2.0, 3.0],
            "d_z_brakiem": [1.0, np.nan, 3.0, np.nan, 5.0],
            "pozycja": ["DEF", "MID", "DEF", "FOR", "MID"],
            "liga": ["L1", "L1", "L2", "L2", "L1"],
            "nieuzywana": [9.0, 9.0, 9.0, 9.0, 9.0],
        }
    )


LICZBOWE = ["a", "b", "c", "d_z_brakiem"]
KATEGORYCZNE = ["pozycja", "liga"]


# --- filtr korelacyjny ---


def test_filtr_usuwa_kopie_kolumny():
    X = np.array([[1.0, 2.0, 5.0], [2.0, 4.0, 1.0], [3.0, 6.0, 4.0], [4.0, 8.0, 2.0]])
    filtr = FiltrKorelacyjny(prog=0.95).fit(X)
    # kolumna 1 jest wielokrotnoscia kolumny 0, wiec wypada ta pozniejsza
    assert list(filtr.zachowane_) == [True, False, True]
    assert filtr.transform(X).shape == (4, 2)


def test_filtr_wylaczony_przepuszcza_wszystko():
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    filtr = FiltrKorelacyjny(prog=None).fit(X)
    assert filtr.zachowane_.all()
    assert filtr.transform(X).shape == X.shape


def test_filtr_znosi_kolumne_stala():
    """Kolumna stala daje NaN w macierzy korelacji i nie moze wywrocic filtra."""
    X = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    filtr = FiltrKorelacyjny(prog=0.95).fit(X)
    assert filtr.zachowane_.all()


def test_filtr_zachowuje_nazwy_cech():
    """Nazwy sa potrzebne w W4 - bez nich ranking SHAP opisuje numery kolumn."""
    X = np.array([[1.0, 2.0, 5.0], [2.0, 4.0, 1.0], [3.0, 6.0, 4.0]])
    filtr = FiltrKorelacyjny(prog=0.95).fit(X)
    assert list(filtr.get_feature_names_out(["a", "b", "c"])) == ["a", "c"]


def test_filtr_jest_deterministyczny():
    losowy = np.random.default_rng(3)
    X = losowy.normal(size=(200, 12))
    X[:, 5] = X[:, 2] + losowy.normal(scale=0.01, size=200)
    pierwszy = FiltrKorelacyjny(prog=0.95).fit(X).zachowane_
    drugi = FiltrKorelacyjny(prog=0.95).fit(X).zachowane_
    assert list(pierwszy) == list(drugi)


def test_filtr_odrzuca_inna_liczbe_kolumn():
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 5.0]])
    filtr = FiltrKorelacyjny(prog=0.95).fit(X)
    with pytest.raises(ValueError, match="uczyl sie na"):
        filtr.transform(np.zeros((3, 5)))


# --- kodowanie kategorii (D-10) ---


def test_koder_nie_upuszcza_poziomu(ramka: pd.DataFrame):
    """
    Pulapka drop="first" z D-10.

    Upuszczenie pierwszego poziomu rozklada wage kategorii odniesienia na wyraz
    wolny, przez co ranking waznosci cech przestaje byc porownywalny miedzy
    modelami. Wynik W4 zalezy od tego ustawienia, wiec pilnujemy go testem.
    """
    przetwarzanie = zbuduj_przetwarzanie(LICZBOWE, KATEGORYCZNE)
    # transformers to lista trojek (nazwa, transformator, kolumny)
    koder = next(
        t for nazwa, t, _ in przetwarzanie.transformers if nazwa == "kategoryczne"
    )
    assert isinstance(koder, OneHotEncoder)
    assert koder.drop is None
    assert koder.handle_unknown == "ignore"

    przetwarzanie.fit(ramka)
    nazwy = list(przetwarzanie.get_feature_names_out())
    # trzy pozycje i dwie ligi to piec kolumn, bez zadnego poziomu odniesienia
    assert sum(n.startswith("pozycja_") for n in nazwy) == 3
    assert sum(n.startswith("liga_") for n in nazwy) == 2


def test_nieznana_kategoria_nie_wywraca_przetwarzania(ramka: pd.DataFrame):
    """Primeira Liga bedzie w E11 kategoria nieznana - ma dac zera, nie wyjatek."""
    przetwarzanie = zbuduj_przetwarzanie(LICZBOWE, KATEGORYCZNE).fit(ramka)
    nowa = ramka.iloc[[0]].copy()
    nowa["liga"] = "Primeira_Liga"

    wynik = przetwarzanie.transform(nowa)
    nazwy = list(przetwarzanie.get_feature_names_out())
    kolumny_ligi = [i for i, n in enumerate(nazwy) if n.startswith("liga_")]
    assert wynik[0, kolumny_ligi].sum() == 0.0


# --- flaga braku (O-3) ---


def test_imputacja_dokleja_flage_braku(ramka: pd.DataFrame):
    """
    Braki procentowe sa strukturalne i skoncentrowane pozycyjnie.

    Sama imputacja mediana wprowadzalaby obciazenie grupowe, wiec obok wartosci
    wstawionej ma isc kolumna mowiaca, ze wartosci tam nie bylo.
    """
    przetwarzanie = zbuduj_przetwarzanie(LICZBOWE, KATEGORYCZNE).fit(ramka)
    nazwy = list(przetwarzanie.get_feature_names_out())
    flagi = [n for n in nazwy if "missingindicator" in n.lower()]
    assert len(flagi) == 1, f"oczekiwano jednej flagi braku, sa: {flagi}"

    # Flaga przechodzi przez StandardScaler razem z reszta, wiec nie ma juz
    # wartosci 0 i 1. Sprawdzamy to, co ma znaczenie: rozdziela wiersze z brakiem
    # od pozostalych i przyjmuje dokladnie dwie wartosci.
    wynik = przetwarzanie.transform(ramka)
    kolumna = wynik[:, nazwy.index(flagi[0])]
    z_brakiem = [1, 3]
    bez_braku = [0, 2, 4]
    assert len(np.unique(kolumna)) == 2
    assert np.allclose(kolumna[z_brakiem], kolumna[1])
    assert np.allclose(kolumna[bez_braku], kolumna[0])
    assert kolumna[1] != kolumna[0]


def test_brak_zostaje_zastapiony_mediana(ramka: pd.DataFrame):
    przetwarzanie = zbuduj_przetwarzanie(LICZBOWE, KATEGORYCZNE).fit(ramka)
    wynik = przetwarzanie.transform(ramka)
    assert np.isfinite(wynik).all()


# --- zakres kolumn ---


def test_kolumna_niewymieniona_nie_wchodzi_do_modelu(ramka: pd.DataFrame):
    """
    remainder="drop": do modelu wchodzi wylacznie to, co ktos wymienil z nazwy.

    Klucze, metadane i zmienna celu leza w tym samym pliku, wiec domyslne
    przepuszczanie reszty konczyloby sie wyciekiem celu do cech.
    """
    przetwarzanie = zbuduj_przetwarzanie(LICZBOWE, KATEGORYCZNE).fit(ramka)
    assert "nieuzywana" not in list(przetwarzanie.get_feature_names_out())
