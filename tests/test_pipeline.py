"""Testy wspolnego przetwarzania cech: filtr korelacyjny, one-hot i flaga braku."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from trustml.models import FiltrKorelacyjny, zbuduj_przetwarzanie


@pytest.fixture
def ramka() -> pd.DataFrame:
    """4 kolumny liczbowe, 2 kategoryczne i jedna, ktorej nie ma w zadnej liscie."""
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            # b = 2 * a
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
    # kolumna 1 = 2 * kolumna 0, wypada pozniejsza
    assert list(filtr.zachowane_) == [True, False, True]
    assert filtr.transform(X).shape == (4, 2)


def test_filtr_wylaczony_przepuszcza_wszystko():
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    filtr = FiltrKorelacyjny(prog=None).fit(X)
    assert filtr.zachowane_.all()
    assert filtr.transform(X).shape == X.shape


def test_filtr_znosi_kolumne_stala():
    """Kolumna stala (NaN w macierzy korelacji) nie psuje filtra."""
    X = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    filtr = FiltrKorelacyjny(prog=0.95).fit(X)
    assert filtr.zachowane_.all()


def test_filtr_zachowuje_nazwy_cech():
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
    OneHotEncoder bez drop (D-10).

    Z drop="first" nieznana liga (same zera) wygladalaby jak kategoria usunieta.
    """
    przetwarzanie = zbuduj_przetwarzanie(LICZBOWE, KATEGORYCZNE)
    koder = next(
        t for nazwa, t, _ in przetwarzanie.transformers if nazwa == "kategoryczne"
    )
    assert isinstance(koder, OneHotEncoder)
    assert koder.drop is None
    assert koder.handle_unknown == "ignore"

    przetwarzanie.fit(ramka)
    nazwy = list(przetwarzanie.get_feature_names_out())
    # 3 pozycje + 2 ligi = 5 kolumn
    assert sum(n.startswith("pozycja_") for n in nazwy) == 3
    assert sum(n.startswith("liga_") for n in nazwy) == 2


def test_nieznana_kategoria_nie_wywraca_przetwarzania(ramka: pd.DataFrame):
    """Nieznana liga (Primeira Liga w E11) daje same zera, a nie wyjatek."""
    przetwarzanie = zbuduj_przetwarzanie(LICZBOWE, KATEGORYCZNE).fit(ramka)
    nowa = ramka.iloc[[0]].copy()
    nowa["liga"] = "Primeira_Liga"

    wynik = przetwarzanie.transform(nowa)
    nazwy = list(przetwarzanie.get_feature_names_out())
    kolumny_ligi = [i for i, n in enumerate(nazwy) if n.startswith("liga_")]
    assert wynik[0, kolumny_ligi].sum() == 0.0


# --- flaga braku (O-3) ---


def test_imputacja_dokleja_flage_braku(ramka: pd.DataFrame):
    """Imputacja dokleja kolumne z flaga braku (D-26)."""
    przetwarzanie = zbuduj_przetwarzanie(LICZBOWE, KATEGORYCZNE).fit(ramka)
    nazwy = list(przetwarzanie.get_feature_names_out())
    flagi = [n for n in nazwy if "missingindicator" in n.lower()]
    assert len(flagi) == 1, f"oczekiwano jednej flagi braku, sa: {flagi}"

    # flaga przechodzi przez StandardScaler, wiec nie jest juz 0/1 - sprawdzam,
    # ze ma dwie wartosci i rozdziela wiersze z brakiem od reszty
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
    """remainder="drop": kolumny spoza list (np. cel) nie wchodza do modelu."""
    przetwarzanie = zbuduj_przetwarzanie(LICZBOWE, KATEGORYCZNE).fit(ramka)
    assert "nieuzywana" not in list(przetwarzanie.get_feature_names_out())
