"""
Analizy wrazliwosci dla etapu E5.

    python -m case_study.football.warianty

Trzy pytania, na kazde jeden przebieg najlepszego modelu z `e5_metryczka.json`.
Hiperparametry biore z tego pliku zamiast stroic od nowa, bo pytanie brzmi
"ile kosztuje ta decyzja", a nie "jak dobry moze byc model".

D-04. Cel nominalny wobec zdeflowanego. Inflacja wycen w Big 5 siega 38 procent
miedzy pierwszym a ostatnim sezonem, a poziom testu lezy 18,4 procent powyzej
treningu, wiec model uczony na cenach nominalnych powinien systematycznie zanizac.
Zeby porownanie bylo uczciwe, predykcje modelu zdeflowanego wracaja do euro
nominalnych przez wspolczynnik sezonu i dopiero tam sa oceniane. Bez tego kroku
porownywalibysmy dwie rozne zmienne celu.

D-25. Koszt usuniecia pary wskaznikow dryblingu. Przebieg z tymi cechami
przywroconymi pokazuje, ile dokladnosci oddajemy za brak przesuniecia definicji
na granicy podzialu. Dolaczamy do niego ceche graniczna, ktora w zbiorze zostala.

O-7. Koszt filtra korelacyjnego przy progu 0,95. Druga polowa bilansu, czyli
stabilnosc rankingu SHAP, domyka sie w W4 - tutaj mierzymy wylacznie to, co filtr
zabiera po stronie dokladnosci.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "case_study.football"

from trustml.evaluation import ocen, wspolczynnik_duana
from trustml.models import zbuduj_model

from .modelowanie import (
    CECHA_GRANICZNA,
    CECHY_ZMIENIONE,
    KATALOG_TABEL,
    kolumny_cech,
    wczytaj,
    zbuduj_estymator,
)
from .parametry import KATALOG_REPO

PROG_FILTRA = 0.95


def wspolczynniki_deflacji(dane: pd.DataFrame) -> np.ndarray:
    """
    Odtwarza wspolczynnik indeksu inflacji dla kazdego wiersza.

    Etap 9 pipeline'u liczyl `market_value_real = market_value_in_eur / wspolczynnik`,
    wiec iloraz obu kolumn oddaje wspolczynnik dokladnie. Indeks powstal wylacznie
    na sezonach treningowych, wiec sciagniecie go stad nie jest wyciekiem.
    """
    return (dane["market_value_in_eur"] / dane["market_value_real"]).to_numpy()


def wynik_wariantu(
    nazwa: str,
    opis: str,
    model,
    dane: pd.DataFrame,
    X_all: pd.DataFrame,
    cel: str,
    osie: list[str],
    *,
    przelicz_na_nominalny: bool = False,
) -> pd.DataFrame:
    """
    Uczy podany model na treningu i ocenia go na kalibracji oraz tescie.

    Przyjmuje:
        nazwa                 - etykieta wariantu
        opis                  - zdanie do kolumny "opis"
        model                 - niedopasowany Pipeline
        dane                  - caly zbior modelowy
        X_all                 - te same wiersze, ograniczone do kolumn cech
        cel                   - kolumna celu, na ktorej model sie uczy
        osie                  - osie stratyfikacji
        przelicz_na_nominalny - czy predykcje wracaja z celu zdeflowanego
                                do euro nominalnych przed ocena

    Zwraca:
        ramke o ukladzie zgodnym z e5_w1.csv.
    """
    trening = (dane["podzial"] == "trening").to_numpy()
    model.fit(X_all[trening], dane.loc[trening, cel].to_numpy())

    wspolczynniki = wspolczynniki_deflacji(dane)

    def predykcja_nominalna(maska) -> np.ndarray:
        surowa = model.predict(X_all[maska])
        if not przelicz_na_nominalny:
            return surowa
        # z log1p realnych euro na realne euro, przez wspolczynnik sezonu
        # na euro nominalne i z powrotem na skale, w ktorej liczymy metryki
        return np.log1p(np.expm1(surowa) * wspolczynniki[maska])

    duan = wspolczynnik_duana(
        dane.loc[trening, "log_market_value"].to_numpy() - predykcja_nominalna(trening)
    )

    czesci = []
    for zbior in ["kalibracja", "test"]:
        maska = (dane["podzial"] == zbior).to_numpy()
        podzbior = dane[maska]
        wynik = ocen(
            podzbior["log_market_value"].to_numpy(),
            predykcja_nominalna(maska),
            {os: podzbior[os].to_numpy() for os in osie},
            duan=duan,
            etykieta=nazwa,
        )
        wynik.insert(0, "zbior", zbior)
        wynik["opis"] = opis
        czesci.append(wynik)
    return pd.concat(czesci, ignore_index=True)


def main() -> None:
    """Liczy trzy warianty najlepszego modelu i zapisuje porownanie."""
    metryczka = json.loads(
        (KATALOG_TABEL / "e5_metryczka.json").read_text(encoding="utf-8")
    )
    podstawa = pd.read_csv(KATALOG_TABEL / "e5_w1.csv")

    # Najlepszy model wybieramy po RMSLE na kalibracji - test zostaje nietkniety
    # do ostatniego pomiaru.
    calosc = podstawa[
        (podstawa["zbior"] == "kalibracja") & (podstawa["os"] == "calosc")
    ]
    najlepszy = calosc.loc[calosc["rmsle"].idxmin(), "model"]
    parametry = metryczka["najlepsze_hiperparametry"][najlepszy]

    print("=" * 78)
    print(f"ANALIZY WRAZLIWOSCI  (etap E5, model {najlepszy})")
    print("=" * 78)
    print(f"\n  hiperparametry z e5_metryczka.json: {parametry}")

    dane, konfiguracja = wczytaj()
    osie = konfiguracja["osie_stratyfikacji"]
    liczbowe, kategoryczne = kolumny_cech(dane)
    X_all = dane[liczbowe + kategoryczne]

    warianty = []

    # Punkt odniesienia dla tej tabeli: ten sam model, ten sam cel, bez zmian.
    warianty.append(
        wynik_wariantu(
            f"{najlepszy}_odniesienie",
            "konfiguracja z E5",
            zbuduj_model(
                zbuduj_estymator(najlepszy, parametry), liczbowe, kategoryczne
            ),
            dane,
            X_all,
            "log_market_value",
            osie,
        )
    )

    # D-04: cel zdeflowany, predykcje przeliczone z powrotem na euro nominalne.
    warianty.append(
        wynik_wariantu(
            f"{najlepszy}_cel_zdeflowany",
            "cel realny, predykcja przeliczona na nominalna (D-04)",
            zbuduj_model(
                zbuduj_estymator(najlepszy, parametry), liczbowe, kategoryczne
            ),
            dane,
            X_all,
            "log_market_value_real",
            osie,
            przelicz_na_nominalny=True,
        )
    )

    # D-25: para dryblingu przywrocona, razem z cecha graniczna.
    liczbowe_z_para = liczbowe + CECHY_ZMIENIONE
    warianty.append(
        wynik_wariantu(
            f"{najlepszy}_z_para_dryblingu",
            "przywrocone " + ", ".join(CECHY_ZMIENIONE) + " (D-25)",
            zbuduj_model(
                zbuduj_estymator(najlepszy, parametry),
                liczbowe_z_para,
                kategoryczne,
            ),
            dane,
            dane[liczbowe_z_para + kategoryczne],
            "log_market_value",
            osie,
        )
    )

    # O-7: filtr korelacyjny przy progu 0,95.
    warianty.append(
        wynik_wariantu(
            f"{najlepszy}_filtr_{PROG_FILTRA}",
            f"filtr korelacyjny przy progu {PROG_FILTRA} (O-7)",
            zbuduj_model(
                zbuduj_estymator(najlepszy, parametry),
                liczbowe,
                kategoryczne,
                prog_korelacji=PROG_FILTRA,
            ),
            dane,
            X_all,
            "log_market_value",
            osie,
        )
    )

    wyniki = pd.concat(warianty, ignore_index=True)
    sciezka = KATALOG_TABEL / "e5_warianty.csv"
    wyniki.to_csv(sciezka, index=False)

    print(f"\n  cecha graniczna zostajaca w zbiorze: {CECHA_GRANICZNA}")
    print("\n  wynik na tescie:\n")
    print(f"    {'wariant':32s} {'RMSLE':>7s} {'MdAPE':>7s} {'R2':>8s} {'agregat':>8s}")
    for _, w in wyniki[
        (wyniki["zbior"] == "test") & (wyniki["os"] == "calosc")
    ].iterrows():
        print(
            f"    {w['model']:32s} {w['rmsle']:7.4f} {w['mdape']:6.1f}% "
            f"{w['r2_log']:8.4f} {w['agregat']:8.2f}"
        )

    print(f"\n  zapisano {len(wyniki)} wierszy do {sciezka.relative_to(KATALOG_REPO)}")


if __name__ == "__main__":
    main()
