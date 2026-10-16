"""
Etap E4: poziomy odniesienia M0a-M0c na kalibracji i tescie.

    python -m case_study.football.poziom_odniesienia

Zapisuje e4_baseline.csv, e4_przeciecia.csv i e4_foldy.csv. Zbior C pominiety (D-14).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "case_study.football"

from trustml.evaluation import (
    mediana_globalna,
    mediana_komorki,
    ocen,
    podsumowanie_foldow,
    przepisanie_z_poprzedniego,
    raport_przeciec,
    sprawdz_odcisk_danych,
    uzupelnij,
    wczytaj_konfiguracje,
    wspolczynnik_duana,
)

from .parametry import KATALOG_REPO, KATALOG_WYJSCIA

SCIEZKA_MODELU = KATALOG_WYJSCIA / "model.parquet"
SCIEZKA_KONFIGURACJI = KATALOG_REPO / "configs" / "split.yaml"
KATALOG_TABEL = KATALOG_REPO / "reports" / "tables"

ZBIORY_OCENIANE = ("kalibracja", "test")


def poprzedni_sezon(sezon: str) -> str:
    """
    "2023-2024" -> "2022-2023". Liczone z roku, nie z pozycji na liscie,
    zeby po 2020-2021 nie wziac 2018-2019 (brak sezonu 2019-2020).
    """
    poczatek = int(str(sezon)[:4])
    return f"{poczatek - 1}-{poczatek}"


def wczytaj() -> tuple[pd.DataFrame, dict]:
    """Wczytuje zbior modelowy i split.yaml, sprawdzajac odcisk danych."""
    konfiguracja = wczytaj_konfiguracje(SCIEZKA_KONFIGURACJI)
    _, odcisk = sprawdz_odcisk_danych(konfiguracja, SCIEZKA_MODELU)
    dane = pd.read_parquet(SCIEZKA_MODELU)

    print(f"  model.parquet {dane.shape[0]} x {dane.shape[1]}")
    print(f"  odcisk zgodny z konfiguracja: {odcisk[:12]}...")
    return dane, konfiguracja


def zbuduj_predykcje(
    dane: pd.DataFrame,
    oceniany: str,
    konfiguracja: dict,
) -> list[dict]:
    """
    Predykcje M0a, M0b, M0c i M0c_pelny dla zbioru `oceniany`.

    Predykcje na treningu sa potrzebne do wspolczynnika Duana (D-21).
    """
    cel = konfiguracja["cel"]["glowny"]
    kolumny_komorki = konfiguracja["poziomy_odniesienia"]["M0b"]["kolumny"]
    klucz, okres = konfiguracja["podzial_zewnetrzny"]["klucz_obserwacji"]

    trening = dane[dane["podzial"] == "trening"]
    cele = dane[dane["podzial"] == oceniany]

    # M0c i tak siega tylko po sezon t-1, wiec historia moze obejmowac wszystko poza C
    historia = dane[dane["podzial"] != "zbior_C"]

    m0a = mediana_globalna(trening, cele, cel)
    m0a_tr = mediana_globalna(trening, trening, cel)

    m0b, uzupelnione_m0b = mediana_komorki(trening, cele, cel, kolumny_komorki)
    m0b_tr, _ = mediana_komorki(trening, trening, cel, kolumny_komorki)

    m0c, pokrycie = przepisanie_z_poprzedniego(
        historia,
        cele,
        klucz=klucz,
        okres=okres,
        cel=cel,
        poprzedni_okres=poprzedni_sezon,
    )
    m0c_tr, pokrycie_tr = przepisanie_z_poprzedniego(
        historia,
        trening,
        klucz=klucz,
        okres=okres,
        cel=cel,
        poprzedni_okres=poprzedni_sezon,
    )
    m0c_pelny, uzupelnione_m0c = uzupelnij(m0c, m0b)

    wszystkie = np.ones(len(cele), dtype=bool)
    return [
        {
            "nazwa": "M0a",
            "opis": "mediana globalna",
            "pred": m0a,
            "pred_trening": m0a_tr,
            "maska_treningu": np.ones(len(trening), dtype=bool),
            "maska": wszystkie,
            "uzupelnione": 0.0,
        },
        {
            "nazwa": "M0b",
            "opis": "mediana w komorce " + " x ".join(kolumny_komorki),
            "pred": m0b,
            "pred_trening": m0b_tr,
            "maska_treningu": np.ones(len(trening), dtype=bool),
            "maska": wszystkie,
            "uzupelnione": float(uzupelnione_m0b.mean()),
        },
        {
            "nazwa": "M0c",
            "opis": "wycena z sezonu t-1, podzbior z pokryciem",
            "pred": m0c[pokrycie],
            "pred_trening": m0c_tr[pokrycie_tr],
            "maska_treningu": pokrycie_tr,
            "maska": pokrycie,
            "uzupelnione": 0.0,
        },
        {
            "nazwa": "M0c_pelny",
            "opis": "wycena z sezonu t-1, braki uzupelnione M0b",
            "pred": m0c_pelny,
            "pred_trening": m0c_tr[pokrycie_tr],
            "maska_treningu": pokrycie_tr,
            "maska": wszystkie,
            "uzupelnione": float(uzupelnione_m0c.mean()),
        },
    ]


def ocen_zbior(
    dane: pd.DataFrame,
    oceniany: str,
    konfiguracja: dict,
) -> pd.DataFrame:
    """Metryki wszystkich poziomow odniesienia na jednym zbiorze."""
    cel = konfiguracja["cel"]["glowny"]
    osie = konfiguracja["osie_stratyfikacji"]

    trening = dane[dane["podzial"] == "trening"]
    cele = dane[dane["podzial"] == oceniany]

    print(f"\n  zbior {oceniany}: {len(cele)} par")

    czesci = []
    for poziom in zbuduj_predykcje(dane, oceniany, konfiguracja):
        podzbior = cele[poziom["maska"]]
        trening_poziomu = trening[poziom["maska_treningu"]]

        # kazdy poziom ma wlasny wspolczynnik Duana z wlasnych reszt treningowych
        duan = wspolczynnik_duana(
            trening_poziomu[cel].to_numpy() - poziom["pred_trening"]
        )

        wynik = ocen(
            podzbior[cel].to_numpy(),
            poziom["pred"],
            {os: podzbior[os].to_numpy() for os in osie},
            duan=duan,
            etykieta=poziom["nazwa"],
        )
        wynik.insert(0, "zbior", oceniany)
        wynik["opis"] = poziom["opis"]
        wynik["pokrycie"] = len(podzbior) / len(cele)
        wynik["uzupelnione"] = poziom["uzupelnione"]
        czesci.append(wynik)

        calosc = wynik.iloc[0]
        print(
            f"    {poziom['nazwa']:10s} pokrycie {100 * wynik['pokrycie'].iloc[0]:5.1f}%"
            f"  RMSLE {calosc['rmsle']:.3f}"
            f"  MdAPE {calosc['mdape']:5.1f}%"
            f"  R2 {calosc['r2_log']:6.3f}"
            f"  MAE {calosc['mae_eur'] / 1e6:5.2f} mln"
            f"  agregat {calosc['agregat']:.2f} -> {calosc['agregat_po_korekcie']:.2f}"
        )

    return pd.concat(czesci, ignore_index=True)


def diagnostyka_podzialu(dane: pd.DataFrame, konfiguracja: dict) -> None:
    """Zapisuje tabele przeciec miedzy zbiorami i opis foldow GroupKFold."""
    klucz = konfiguracja["podzial_zewnetrzny"]["klucz_obserwacji"]
    grupa = konfiguracja["podzial_wewnetrzny"]["kolumna_grupujaca"]
    n_podzialow = konfiguracja["podzial_wewnetrzny"]["n_podzialow"]

    przeciecia = raport_przeciec(dane, klucz=klucz, grupa=grupa)
    przeciecia.to_csv(KATALOG_TABEL / "e4_przeciecia.csv", index=False)
    print("\n  przeciecia miedzy zbiorami (wspolne pary musza byc zerem):")
    for _, w in przeciecia.iterrows():
        print(
            f"    {w['zbior_a']:11s} x {w['zbior_b']:11s}"
            f"  par {w['wspolnych_par']:3d}"
            f"  zawodnikow {w['wspolnych_grup']:5d}"
            f"  ({100 * w['udzial_grup_b']:4.1f}% skladu drugiego zbioru)"
        )

    trening = dane[dane["podzial"] == "trening"]
    foldy = podsumowanie_foldow(trening[grupa].to_numpy(), n_podzialow)
    foldy.to_csv(KATALOG_TABEL / "e4_foldy.csv", index=False)
    print(f"\n  GroupKFold po {grupa}, {n_podzialow} czesci na treningu:")
    for _, w in foldy.iterrows():
        print(
            f"    fold {int(w['fold'])}  ucz {int(w['n_ucz']):5d} wierszy"
            f"  wal {int(w['n_wal']):5d} wierszy"
            f"  ({int(w['grup_wal']):4d} zawodnikow,"
            f" wspolnych z uczaca: {int(w['grup_wspolnych'])})"
        )


def main() -> None:
    """Liczy caly etap i zapisuje tabele do reports/tables/."""
    print("=" * 78)
    print("POZIOM ODNIESIENIA I PROTOKOL WALIDACJI  (etap E4)")
    print("=" * 78)
    print()

    KATALOG_TABEL.mkdir(parents=True, exist_ok=True)
    dane, konfiguracja = wczytaj()

    diagnostyka_podzialu(dane, konfiguracja)

    wyniki = pd.concat(
        [ocen_zbior(dane, zbior, konfiguracja) for zbior in ZBIORY_OCENIANE],
        ignore_index=True,
    )
    sciezka = KATALOG_TABEL / "e4_baseline.csv"
    wyniki.to_csv(sciezka, index=False)

    print(f"\n  zapisano {len(wyniki)} wierszy do {sciezka.relative_to(KATALOG_REPO)}")

    test = wyniki[(wyniki["zbior"] == "test") & (wyniki["os"] == "znany_z_treningu")]
    if not test.empty:
        print("\n  test, os znani / nowi:")
        for _, w in test.iterrows():
            print(
                f"    {w['model']:10s} {'znani' if w['grupa'] == 'True' else 'nowi':6s}"
                f"  n {int(w['n']):5d}  RMSLE {w['rmsle']:.3f}  R2 {w['r2_log']:6.3f}"
            )


if __name__ == "__main__":
    main()
