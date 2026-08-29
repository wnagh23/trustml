"""
Testy metryk i poziomow odniesienia (etap E4).

    pytest tests/test_metryki.py -v

Metryki sa tu jedynym narzedziem pomiarowym calej pracy, wiec sprawdzamy je na
przykladach o znanym wyniku. Najwazniejszy jest test korekty Duana: sprawdza, ze
po powrocie ze skali logarytmicznej srednia predykcji zgadza sie ze srednia
rzeczywista, czego samo `expm1` nie zapewnia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trustml.evaluation import (
    do_euro,
    mdape,
    mediana_globalna,
    mediana_komorki,
    metryki,
    ocen,
    przepisanie_z_poprzedniego,
    r2_log,
    rmsle,
    uzupelnij,
    wspolczynnik_duana,
)


@pytest.fixture
def panel() -> pd.DataFrame:
    """Maly panel: czterech zawodnikow, trzy sezony, wartosci w skali log1p."""
    wiersze = [
        ("a", "2021-2022", "MID", "L1", 10.0),
        ("a", "2022-2023", "MID", "L1", 11.0),
        ("a", "2023-2024", "MID", "L1", 12.0),
        ("b", "2021-2022", "DEF", "L1", 14.0),
        ("b", "2023-2024", "DEF", "L1", 15.0),
        ("c", "2022-2023", "MID", "L2", 16.0),
        ("c", "2023-2024", "MID", "L2", 17.0),
        ("d", "2023-2024", "DEF", "L2", 13.0),
    ]
    return pd.DataFrame(
        wiersze, columns=["player_id", "season", "pozycja", "liga", "y"]
    )


def poprzedni(sezon: str) -> str:
    poczatek = int(sezon[:4])
    return f"{poczatek - 1}-{poczatek}"


# --- metryki ---


def test_idealna_predykcja():
    y = np.array([10.0, 12.0, 14.0])
    assert rmsle(y, y) == pytest.approx(0.0)
    assert r2_log(y, y) == pytest.approx(1.0)
    assert mdape(y, y) == pytest.approx(0.0)


def test_rmsle_to_rmse_w_skali_logarytmicznej():
    y = np.array([10.0, 10.0, 10.0])
    pred = np.array([11.0, 9.0, 10.0])
    # bledy 1, -1, 0 -> pierwiastek ze sredniej 2/3
    assert rmsle(y, pred) == pytest.approx(np.sqrt(2 / 3))


def test_mdape_na_znanym_przykladzie():
    """Predykcja dwa razy wyzsza od prawdy daje blad wzgledny 100 procent."""
    prawda = np.array([1_000_000.0, 4_000_000.0, 2_000_000.0])
    y = np.log1p(prawda)
    pred = np.log(2 * prawda)  # log, bo do_euro odejmuje jedynke z powrotem
    assert mdape(y, pred) == pytest.approx(100.0, rel=1e-4)


def test_r2_dla_predykcji_stalej_jest_zerem():
    y = np.array([10.0, 12.0, 14.0, 16.0])
    assert r2_log(y, np.full(4, y.mean())) == pytest.approx(0.0)


def test_brak_w_predykcji_jest_bledem():
    y = np.array([10.0, 12.0])
    with pytest.raises(ValueError, match="brak w predykcji"):
        rmsle(y, np.array([10.0, np.nan]))


def test_rozne_dlugosci_sa_bledem():
    with pytest.raises(ValueError, match="rozne dlugosci"):
        rmsle(np.array([1.0, 2.0]), np.array([1.0]))


# --- korekta Duana ---


def test_duan_bez_reszt_jest_neutralny():
    assert wspolczynnik_duana(np.zeros(10)) == pytest.approx(1.0)


def test_do_euro_bez_korekty_to_expm1():
    pred = np.array([10.0, 15.0])
    assert do_euro(pred) == pytest.approx(np.expm1(pred))


def test_duan_odzyskuje_srednia_w_euro():
    """
    Sedno D-21: exp(sredniej z logarytmow) daje srednia geometryczna.

    Budujemy dane, w ktorych prawda jest rozkladem lognormalnym wokol stalej
    predykcji. Bez korekty srednia predykcji lezy wyraznie ponizej sredniej
    rzeczywistej; po korekcie obie sie schodza.
    """
    losowy = np.random.default_rng(7)
    srodek, rozrzut, n = 15.0, 0.8, 200_000
    y = srodek + losowy.normal(0.0, rozrzut, n)
    pred = np.full(n, srodek)

    srednia_prawdy = np.expm1(y).mean()
    bez_korekty = do_euro(pred).mean()
    duan = wspolczynnik_duana(y - pred)

    # teoretyczna wartosc wspolczynnika dla rozkladu normalnego reszt
    assert duan == pytest.approx(np.exp(rozrzut**2 / 2), rel=0.01)
    assert bez_korekty < 0.85 * srednia_prawdy
    assert do_euro(pred, duan).mean() == pytest.approx(srednia_prawdy, rel=0.01)


def test_korekta_nie_dotyka_miar_logarytmicznych():
    """RMSLE i MdAPE licza sie na predykcjach surowych, wiec duan ich nie rusza."""
    y = np.array([10.0, 12.0, 14.0])
    pred = np.array([10.5, 11.0, 14.5])
    bez = metryki(y, pred, duan=1.0)
    z_korekta = metryki(y, pred, duan=1.3)
    assert bez["rmsle"] == z_korekta["rmsle"]
    assert bez["mdape"] == z_korekta["mdape"]
    assert bez["r2_log"] == z_korekta["r2_log"]
    assert bez["mae_eur"] != z_korekta["mae_eur"]


# --- stratyfikacja ---


def test_ocen_zaczyna_od_calosci_i_dzieli_na_grupy():
    y = np.array([10.0, 12.0, 14.0, 16.0])
    pred = y + 0.1
    grupy = {"pozycja": np.array(["DEF", "DEF", "MID", "MID"])}

    wynik = ocen(y, pred, grupy, etykieta="M0x")
    assert list(wynik["os"]) == ["calosc", "pozycja", "pozycja"]
    assert wynik.loc[0, "n"] == 4
    assert wynik["n"].iloc[1:].sum() == 4
    assert set(wynik["model"]) == {"M0x"}


def test_ocen_pilnuje_dlugosci_osi():
    y = np.array([10.0, 12.0])
    with pytest.raises(ValueError, match="wartosci wobec"):
        ocen(y, y, {"pozycja": np.array(["DEF"])})


# --- poziomy odniesienia ---


def test_mediana_globalna_jest_stala(panel: pd.DataFrame):
    trening = panel[panel["season"] != "2023-2024"]
    pred = mediana_globalna(trening, panel, "y")
    assert len(pred) == len(panel)
    assert np.unique(pred).size == 1
    assert pred[0] == pytest.approx(trening["y"].median())


def test_mediana_komorki_uzupelnia_nieznane_komorki(panel: pd.DataFrame):
    """Komorka nieobecna w treningu dostaje mediane globalna i zostaje oznaczona."""
    trening = panel[panel["season"] != "2023-2024"]
    pred, uzupelnione = mediana_komorki(trening, panel, "y", ["pozycja", "liga"])

    # DEF x L2 wystepuje wylacznie w sezonie testowym
    brakujaca = (panel["pozycja"] == "DEF") & (panel["liga"] == "L2")
    assert uzupelnione[brakujaca.to_numpy()].all()
    assert (~uzupelnione[~brakujaca.to_numpy()]).all()
    assert pred[brakujaca.to_numpy()][0] == pytest.approx(trening["y"].median())

    # MID x L1 ma w treningu dwie obserwacje: 10 i 11
    mid_l1 = ((panel["pozycja"] == "MID") & (panel["liga"] == "L1")).to_numpy()
    assert pred[mid_l1][0] == pytest.approx(10.5)


def test_przepisanie_bierze_wartosc_z_poprzedniego_sezonu(panel: pd.DataFrame):
    cele = panel[panel["season"] == "2023-2024"]
    pred, pokrycie = przepisanie_z_poprzedniego(
        panel,
        cele,
        klucz="player_id",
        okres="season",
        cel="y",
        poprzedni_okres=poprzedni,
    )
    # a i c maja sezon 2022-2023; b ma luke, d pojawia sie po raz pierwszy
    assert list(pokrycie) == [True, False, True, False]
    assert pred[0] == pytest.approx(11.0)
    assert pred[2] == pytest.approx(16.0)


def test_przepisanie_odrzuca_zduplikowana_historie(panel: pd.DataFrame):
    with pytest.raises(ValueError, match="powtorzone pary"):
        przepisanie_z_poprzedniego(
            pd.concat([panel, panel]),
            panel,
            klucz="player_id",
            okres="season",
            cel="y",
            poprzedni_okres=poprzedni,
        )


def test_uzupelnienie_lata_dziury(panel: pd.DataFrame):
    z_dziurami = np.array([1.0, np.nan, 3.0])
    zapas = np.array([9.0, 9.0, 9.0])
    pelne, uzupelnione = uzupelnij(z_dziurami, zapas)
    assert list(pelne) == [1.0, 9.0, 3.0]
    assert list(uzupelnione) == [False, True, False]
    # oryginal zostaje nietkniety
    assert np.isnan(z_dziurami[1])
