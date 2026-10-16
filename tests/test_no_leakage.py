"""
Testy podzialu: brak wspolnych par zawodnik-sezon miedzy zbiorami, zgodnosc
danych z split.yaml i GroupKFold bez dzielenia zawodnika miedzy foldy.
"""

from __future__ import annotations

import pandas as pd
import pytest

from case_study.football.parametry import KATALOG_REPO, KATALOG_WYJSCIA
from case_study.football.poziom_odniesienia import poprzedni_sezon
from trustml.evaluation import (
    pary,
    podzial_grupowy,
    raport_przeciec,
    sprawdz_odcisk_danych,
    wczytaj_konfiguracje,
)

SCIEZKA_MODELU = KATALOG_WYJSCIA / "model.parquet"
SCIEZKA_KONFIGURACJI = KATALOG_REPO / "configs" / "split.yaml"


@pytest.fixture(scope="module")
def konfiguracja() -> dict:
    return wczytaj_konfiguracje(SCIEZKA_KONFIGURACJI)


@pytest.fixture(scope="module")
def model() -> pd.DataFrame:
    if not SCIEZKA_MODELU.exists():
        pytest.skip(
            f"brak {SCIEZKA_MODELU} - uruchom "
            f"python -m case_study.football.przygotuj_dane"
        )
    return pd.read_parquet(SCIEZKA_MODELU)


# --- zgodnosc konfiguracji z danymi ---


def test_odcisk_danych_zgadza_sie_z_konfiguracja(konfiguracja: dict):
    """
    SHA-256 model.parquet zgadza sie z split.yaml.

    Po zmianie danych trzeba recznie wpisac nowy odcisk i przeliczyc wyniki.
    """
    if not SCIEZKA_MODELU.exists():
        pytest.skip("brak model.parquet")
    zgodny, faktyczny = sprawdz_odcisk_danych(
        konfiguracja, SCIEZKA_MODELU, twardo=False
    )
    assert zgodny, (
        f"model.parquet ma odcisk {faktyczny}, a split.yaml opisuje "
        f"{konfiguracja['dane']['sha256']}"
    )


def test_liczebnosci_zgodne_z_konfiguracja(model: pd.DataFrame, konfiguracja: dict):
    for nazwa, opis in konfiguracja["podzial_zewnetrzny"]["zbiory"].items():
        assert (model["podzial"] == nazwa).sum() == opis["wiersze"], nazwa


def test_sezony_zgodne_z_konfiguracja(model: pd.DataFrame, konfiguracja: dict):
    for nazwa, opis in konfiguracja["podzial_zewnetrzny"]["zbiory"].items():
        w_danych = set(model.loc[model["podzial"] == nazwa, "season"])
        assert w_danych == set(opis["sezony"]), nazwa


def test_liga_odlozona_wystepuje_tylko_w_zbiorze_c(
    model: pd.DataFrame, konfiguracja: dict
):
    liga = konfiguracja["podzial_zewnetrzny"]["liga_wylacznie_w_zbiorze_c"]
    assert set(model.loc[model["liga"] == liga, "podzial"]) == {"zbior_C"}


def test_brakujacy_sezon_nie_wrocil(model: pd.DataFrame, konfiguracja: dict):
    """Sezon 2019-2020 ma byc nieobecny."""
    assert konfiguracja["podzial_zewnetrzny"]["brakujacy_sezon"] not in set(
        model["season"]
    )


# --- brak wycieku miedzy zbiorami ---


def test_pary_nie_powtarzaja_sie_miedzy_zbiorami(
    model: pd.DataFrame, konfiguracja: dict
):
    klucz = konfiguracja["podzial_zewnetrzny"]["klucz_obserwacji"]
    przeciecia = raport_przeciec(model, klucz=klucz)
    wadliwe = przeciecia[przeciecia["wspolnych_par"] > 0]
    assert wadliwe.empty, f"wspolne pary miedzy zbiorami:\n{wadliwe}"


def test_klucz_pokrywa_caly_zbior(model: pd.DataFrame, konfiguracja: dict):
    klucz = konfiguracja["podzial_zewnetrzny"]["klucz_obserwacji"]
    assert len(pary(model, klucz)) == len(model)


def test_wspolni_zawodnicy_sa_udokumentowani(model: pd.DataFrame, konfiguracja: dict):
    """Wspolni zawodnicy trening/test zgadzaja sie z flaga znany_z_treningu (D-12)."""
    klucz = konfiguracja["podzial_zewnetrzny"]["klucz_obserwacji"]
    przeciecia = raport_przeciec(model, klucz=klucz)
    wiersz = przeciecia[
        (przeciecia["zbior_a"] == "test") & (przeciecia["zbior_b"] == "trening")
    ].iloc[0]
    assert wiersz["wspolnych_grup"] > 0, "podzial czasowy bez wspolnych zawodnikow"

    zawodnicy_treningu = set(model.loc[model["podzial"] == "trening", "player_id"])
    test = model[model["podzial"] == "test"]
    assert wiersz["wspolnych_grup"] == test["player_id"].isin(zawodnicy_treningu).sum()
    assert (
        test["znany_z_treningu"] == test["player_id"].isin(zawodnicy_treningu)
    ).all()


# --- podzial wewnetrzny ---


def test_groupkfold_nie_dzieli_zawodnika(model: pd.DataFrame, konfiguracja: dict):
    grupa = konfiguracja["podzial_wewnetrzny"]["kolumna_grupujaca"]
    n = konfiguracja["podzial_wewnetrzny"]["n_podzialow"]
    grupy = model.loc[model["podzial"] == "trening", grupa].to_numpy()

    for ucz, wal in podzial_grupowy(grupy, n):
        assert not (set(grupy[ucz]) & set(grupy[wal]))


def test_groupkfold_pokrywa_trening_dokladnie_raz(
    model: pd.DataFrame, konfiguracja: dict
):
    grupa = konfiguracja["podzial_wewnetrzny"]["kolumna_grupujaca"]
    n = konfiguracja["podzial_wewnetrzny"]["n_podzialow"]
    grupy = model.loc[model["podzial"] == "trening", grupa].to_numpy()

    licznik = {}
    for _, wal in podzial_grupowy(grupy, n):
        for pozycja in wal:
            licznik[pozycja] = licznik.get(pozycja, 0) + 1
    assert len(licznik) == len(grupy)
    assert set(licznik.values()) == {1}


def test_groupkfold_jest_deterministyczny(model: pd.DataFrame, konfiguracja: dict):
    grupa = konfiguracja["podzial_wewnetrzny"]["kolumna_grupujaca"]
    n = konfiguracja["podzial_wewnetrzny"]["n_podzialow"]
    grupy = model.loc[model["podzial"] == "trening", grupa].to_numpy()

    for (a_ucz, a_wal), (b_ucz, b_wal) in zip(
        podzial_grupowy(grupy, n), podzial_grupowy(grupy, n), strict=True
    ):
        assert (a_ucz == b_ucz).all()
        assert (a_wal == b_wal).all()


# --- poziom odniesienia M0c ---


def test_poprzedni_sezon_cofa_o_rok():
    assert poprzedni_sezon("2023-2024") == "2022-2023"
    assert poprzedni_sezon("2020-2021") == "2019-2020"


def test_m0c_siega_wylacznie_w_przeszlosc(model: pd.DataFrame):
    """M0c bierze wycene z sezonu wczesniejszego, nie pozniejszego."""
    for sezon in model.loc[model["podzial"] == "test", "season"].unique():
        assert poprzedni_sezon(sezon) < sezon
