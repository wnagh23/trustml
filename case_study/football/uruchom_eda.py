"""
Uruchomienie pelnej analizy eksploracyjnej (etap E3).

    python -m case_study.football.uruchom_eda

albo po prostu przyciskiem uruchamiania w edytorze - blok startowy nizej
dokleja korzen repozytorium do sciezki importow, wiec oba sposoby dzialaja.

Skrypt przechodzi po kolei przez wszystkie pytania z listy E3 w `04-plan.md`,
zapisuje figury do `reports/figures/eda/`, tabele do `reports/tables/eda/`
i zrzuca komplet zmierzonych liczb do `reports/tables/eda/fakty.json`.

Ten ostatni plik jest tu najwazniejszy: raport `docs/05-eda.md` cytuje liczby
wylacznie stad, wiec kazde zdanie raportu da sie sprawdzic przez porownanie
z zawartoscia JSON-a. Przebieg trwa okolo minuty i czyta wylacznie
`data/processed/`, wiec nie wymaga danych surowych.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "case_study.football"

from . import eda


def main() -> None:
    """Wykonuje wszystkie analizy w ustalonej kolejnosci i zapisuje fakty."""
    start = time.perf_counter()

    print("=" * 78)
    print("ANALIZA EKSPLORACYJNA  (etap E3)")
    print("=" * 78)

    eda.ustaw_styl()
    model, sezony = eda.wczytaj()
    print(
        f"\nwczytano: model.parquet {model.shape[0]} x {model.shape[1]}, "
        f"zawodnik_sezon.parquet {sezony.shape[0]} x {sezony.shape[1]}"
    )
    eda.zapisz_fakt("wierszy_model", len(model))
    eda.zapisz_fakt("kolumn_model", model.shape[1])
    for zbior, liczba in model["podzial"].value_counts().items():
        eda.zapisz_fakt(f"wierszy_{zbior}", int(liczba))

    # Kolejnosc odpowiada kolejnosci pytan w planie E3. Kazda funkcja jest
    # niezalezna od pozostalych, wiec dowolna z nich da sie uruchomic osobno
    # w konsoli przy diagnozowaniu.
    eda.a1_rozklad_celu(model)
    eda.a2_deflacja(model)
    eda.a3_krzywa_wieku(model)
    eda.a4_struktura_panelu(model)
    eda.a5_podgrupy(model)
    eda.a6_korelacje(model)
    eda.a7_mapa_brakow(model)
    eda.a8_prog_minut(model)
    eda.a9_sufit_informacyjny(model)
    eda.a10_dryf_cech(model)
    eda.a11_zlom_definicyjny(model)
    eda.a12_ciekawostki(model, sezony)

    sciezka_faktow = eda.KATALOG_TABEL / "fakty.json"
    # `sort_keys=True` daje stabilna kolejnosc kluczy, wiec kolejne przebiegi
    # roznia sie w gicie wylacznie tam, gdzie zmienily sie liczby.
    # `ensure_ascii=False` zostawia polskie znaki w postaci czytelnej.
    sciezka_faktow.write_text(
        json.dumps(eda.FAKTY, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print(
        f"gotowe w {time.perf_counter() - start:.1f} s, "
        f"{len(eda.FAKTY)} zmierzonych liczb w reports/tables/eda/fakty.json"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
