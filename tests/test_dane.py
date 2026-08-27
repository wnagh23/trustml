"""
Testy zbioru modelowego.

    pytest tests/ -v
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from case_study.football.parametry import (
    KATALOG_WYJSCIA,
    KLUCZ_SEZON,
    LIGA_ZBIOR_C,
    MIN_MINUT,
    SEZONY,
    SEZONY_KALIBRACJA,
    SEZONY_TEST,
    SEZONY_TRENING,
    WSKAZNIKI_PROCENTOWE,
)

SCIEZKA_MODELU = KATALOG_WYJSCIA / "model.parquet"
SCIEZKA_MANIFESTU = KATALOG_WYJSCIA / "manifest.json"

# Liczebnosci z ostatniego zweryfikowanego przebiegu
OCZEKIWANE_WIERSZE = 13_709
OCZEKIWANE_KOLUMNY = 115
OCZEKIWANY_PODZIAL = {
    "trening": 7_924,
    "kalibracja": 1_982,
    "test": 1_957,
    "zbior_C": 1_846,
}


@pytest.fixture(scope="module")
def model() -> pd.DataFrame:
    """Wczytuje zbior modelowy raz na caly modul testow."""
    if not SCIEZKA_MODELU.exists():
        pytest.skip(
            f"brak {SCIEZKA_MODELU} - uruchom "
            f"python -m case_study.football.przygotuj_dane"
        )
    return pd.read_parquet(SCIEZKA_MODELU)


# --- ksztalt zbioru ---


def test_rozmiar(model: pd.DataFrame):
    assert len(model) == OCZEKIWANE_WIERSZE
    assert model.shape[1] == OCZEKIWANE_KOLUMNY


def test_klucz_jest_unikalny(model: pd.DataFrame):
    assert not model.duplicated(subset=KLUCZ_SEZON).any()


def test_liczebnosci_podzialu(model: pd.DataFrame):
    assert model["podzial"].value_counts().to_dict() == OCZEKIWANY_PODZIAL


def test_tylko_wybrane_sezony(model: pd.DataFrame):
    assert set(model["season"]) <= set(SEZONY)


# --- poprawnosc podzialu ---


def test_primeira_liga_tylko_w_zbiorze_c(model: pd.DataFrame):
    """Liga odlozona na pomiar przesuniecia dziedziny zostaje wylacznie w zbiorze C."""
    for zbior in ["trening", "kalibracja", "test"]:
        ligi = set(model.loc[model["podzial"] == zbior, "liga"])
        assert LIGA_ZBIOR_C not in ligi, f"{LIGA_ZBIOR_C} w zbiorze {zbior}"


def test_zbior_c_to_wylacznie_primeira_liga(model: pd.DataFrame):
    assert set(model.loc[model["podzial"] == "zbior_C", "liga"]) == {LIGA_ZBIOR_C}


def test_sezony_zgodne_z_podzialem(model: pd.DataFrame):
    """Kazdy zbior Big 5 obejmuje dokladnie te sezony, ktore mu przypisano."""
    for zbior, oczekiwane in [
        ("trening", SEZONY_TRENING),
        ("kalibracja", SEZONY_KALIBRACJA),
        ("test", SEZONY_TEST),
    ]:
        assert set(model.loc[model["podzial"] == zbior, "season"]) == set(oczekiwane)


def test_pary_sa_rozlaczne_miedzy_zbiorami(model: pd.DataFrame):
    """Kazda para (zawodnik, sezon) nalezy do dokladnie jednego zbioru."""
    zbiory = model["podzial"].unique()
    klucze = {
        z: set(map(tuple, model.loc[model["podzial"] == z, KLUCZ_SEZON].to_numpy()))
        for z in zbiory
    }
    for i, a in enumerate(zbiory):
        for b in zbiory[i + 1 :]:
            assert not (klucze[a] & klucze[b]), f"wspolne pary: {a} i {b}"


def test_flaga_znany_z_treningu(model: pd.DataFrame):
    """
    Flaga musi byc falszywa w treningu i zgodna z faktycznym skladem gdzie indziej.

    Sluzy do stratyfikacji wynikow (D-12): model rozpoznaje zawodnikow, ktorych
    juz widzial, wiec metryki dla znanych i nowych trzeba raportowac osobno.
    """
    trening = model[model["podzial"] == "trening"]
    assert not trening["znany_z_treningu"].any()

    zawodnicy_treningu = set(trening["player_id"])
    reszta = model[model["podzial"] != "trening"]
    oczekiwana = reszta["player_id"].isin(zawodnicy_treningu)
    assert (reszta["znany_z_treningu"] == oczekiwana).all()


# --- wartosci cech i celu ---


def test_prog_minut(model: pd.DataFrame):
    assert (model["minuty_sezon"] >= MIN_MINUT).all()


def test_wskazniki_procentowe_w_zakresie(model: pd.DataFrame):
    """Wskaznik miesci sie w [0, 100] albo jest NaN - brak strukturalny."""
    for kolumna in WSKAZNIKI_PROCENTOWE:
        obecne = model[kolumna].dropna()
        assert obecne.between(0, 100).all(), f"{kolumna} poza zakresem"


def test_cel_jest_dodatni_i_kompletny(model: pd.DataFrame):
    assert model["market_value_in_eur"].gt(0).all()
    assert model["log_market_value"].notna().all()
    assert model["market_value_real"].gt(0).all()


def test_cechy_p90_sa_skonczone(model: pd.DataFrame):
    """Nieskonczonosc w _p90 oznaczalaby dzielenie przez zero minut."""
    kolumny = [c for c in model.columns if c.endswith("_p90")]
    assert kolumny, "brak kolumn _p90"
    import numpy as np

    assert np.isfinite(model[kolumny].to_numpy()).all()


def test_wiek_w_sensownym_zakresie(model: pd.DataFrame):
    assert model["wiek"].between(14, 50).all()
    # wiek_do_kw ma byc dokladnym kwadratem wieku z tego samego wiersza
    assert (model["wiek_do_kw"] == model["wiek"] ** 2).all()


def test_kategoryczne_zostaly_tekstem(model: pd.DataFrame):
    """Kodowanie ma sie dziac w Pipeline, na tekstowych kategoriach ze zbioru."""
    for kolumna in ["pozycja", "region", "liga", "noga"]:
        assert model[kolumna].dtype == object or isinstance(
            model[kolumna].dtype, pd.StringDtype
        ), f"{kolumna} nie jest tekstem"
    assert set(model["pozycja"]) == {"DEF", "MID", "FOR"}
    assert set(model["region"]) <= {"EUROPA", "AMERYKA_PLD", "RESZTA"}


def test_brak_kolumn_zabronionych(model: pd.DataFrame):
    """
    Zbior modelowy zostaje wolny od cech swiadomie wykluczonych.

    """
    assert "market_value_prev" not in model.columns
    assert "attendance" not in model.columns
    # surowe sumy: kolumna licznikowa bez sufiksu _p90, ktora ma odpowiednik _p90
    for kolumna in ["goals", "xG", "total_completed", "minutes"]:
        assert kolumna not in model.columns, f"surowa suma {kolumna} w zbiorze"


# --- manifest ---


def test_manifest_istnieje_i_opisuje_ten_plik(model: pd.DataFrame):
    """Manifest ma dotyczyc tego samego przebiegu, co zapisany model.parquet."""
    if not SCIEZKA_MANIFESTU.exists():
        pytest.skip("brak manifest.json")

    manifest = json.loads(SCIEZKA_MANIFESTU.read_text(encoding="utf-8"))
    assert manifest["wiersze_po_etapach"]["etap_10_zbior_modelowy"] == len(model)

    for nazwa in ["master.db", "players.csv", "player_valuations.csv"]:
        wpis = manifest["pliki_wejsciowe"][nazwa]
        assert not wpis.get("brak"), f"{nazwa} nieopisany w manifescie"
        assert len(wpis["sha256"]) == 64

    assert manifest["parametry"]["MIN_MINUT"] == MIN_MINUT
