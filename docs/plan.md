# Plan i status

| etap | zakres | status |
|---|---|---|
| E1 | pipeline danych | zrobione |
| E2 | manifest, testy, importy | zrobione |
| E3 | analiza eksploracyjna | zrobione |
| E4 | protokół walidacji, poziomy odniesienia | zrobione |
| E5 | modele M1-M3, wymiar W1 | w trakcie |
| E6 | trustml 0.1: szkielet, W1, W6 | do zrobienia |
| E7 | odporność (W2) | do zrobienia |
| E8 | niepewność (W3) | do zrobienia |
| E9 | wyjaśnialność (W4) | do zrobienia |
| E10 | sprawiedliwość (W5) | do zrobienia |
| E11 | przesunięcie dziedziny: Primeira Liga | do zrobienia |
| E12 | trustml 1.0 i podsumowanie | do zrobienia |

## Wymiary

- **W1 dokładność:** RMSLE i MdAPE, R² w skali log pomocniczo (D-15).
- **W2 odporność:** spadek jakości na zbiorze C, rozłożony na przesunięcie i rozrzut (D-13); wrażliwość na szum w cechach.
- **W3 niepewność:** przedziały konforemne (MAPIE) kalibrowane na sezonie 2022-23; pokrycie i szerokość przedziału zawsze razem.
- **W4 wyjaśnialność:** ranking ważności SHAP i jego stabilność między ziarnami i foldami (tau Kendalla).
- **W5 sprawiedliwość:** różnice błędu i średniej reszty między grupami (region, wiek, pozycja, liga).
- **W6 odtwarzalność:** rozrzut wyników między ziarnami i kompletny zapis przebiegu (dane, wersje, parametry).

Hipoteza: rankingi modeli w różnych wymiarach się różnią. Sprawdzę to tau Kendalla między rankingami w E12.

## Zrobione

**E1-E2.** Pipeline w 10 etapach, zbiór 13 709 x 115 (szczegóły w `docs/dane.md`). Manifest z SHA-256 plików wejściowych, testy danych w `tests/test_dane.py`, pre-commit z ruffem.

**E3.** EDA w `case_study/football/eda.py`, wnioski w `docs/dane.md`. Zostało: próg minut 0, wymaga ponownego przebiegu pipeline'u z innym `MIN_MINUT`.

**E4.** `configs/split.yaml` z SHA-256 danych, GroupKFold po zawodniku, testy przecieków w `tests/test_no_leakage.py`, metryki i poziomy odniesienia w `src/trustml/evaluation/`. Na teście: M0c 0,53 RMSLE na 72% zawodników z wyceną sprzed roku, 0,76 na całym teście.

**E5.** Wspólny Pipeline (`src/trustml/models/pipeline.py`), M1 Elastic Net, M2 Random Forest, M3 XGBoost, strojenie Optuną (100 prób, GroupKFold na treningu). Wynik na teście:

| model | RMSLE | R² (log) | MdAPE |
|---|---|---|---|
| M1 Elastic Net | 0,672 | 0,706 | 40,7% |
| M2 Random Forest | 0,700 | 0,680 | 43,4% |
| M3 XGBoost | 0,646 | 0,727 | 38,3% |
| M0c, cały test | 0,761 | 0,622 | 40,0% |

Na zawodnikach nowych (34% testu) M3 ma R² 0,65, na znanych 0,76.

Zostało w E5: uruchomić `python -m case_study.football.warianty` (cel zdeflowany, para dryblingu, filtr korelacyjny).

## Do zrobienia

**E6 (4-6 dni).** Klasa `TrustReport` (model, zbiory treningowy, kalibracyjny, testowy i C, grupy, cechy wrażliwe), moduły W1 i W6, raport HTML, testy powyżej 70% pokrycia, CI na GitHub Actions.

**E7.** Rolling origin (uwaga na brak sezonu 2019-20), szum 1/5/10% na cechach `_p90`, maskowanie cech, nachylenie krzywej spadku jakości.

**E8.** MAPIE, alfa 0,1 i 0,2, pokrycie w grupach i dla znanych/nowych, porównanie z regresją kwantylową i rozrzutem drzew w RF. Podział czasowy łamie założenie wymienialności, trzeba to pokazać liczbowo.

**E9.** SHAP (TreeExplainer, LinearExplainer), permutation importance, PDP/ALE, tau Kendalla dla 10 ziaren.

**E10.** Różnice RMSLE/MdAPE i średniej reszty między grupami, bootstrap, różnica pokrycia W3 między grupami. Wnioski o obciążeniu formułuję o wycenach rynku.

**E11.** Pełny raport na zbiorze C, z ligą i bez (D-11), porównanie nowi vs nowi (D-12), rozkład błędu (D-13), wariant z korektą poziomu na małej próbce z C.

**E12.** Wszystkie wymiary w `TrustReport`, `scripts/run_all.py`, test na drugim zbiorze (ACSIncome z folktables), macierz model x wymiar, tau Kendalla między rankingami, test istotności wobec rozrzutu między ziarnami, wersja 1.0.

## Otwarte pytania

- **O-4:** Czym trustml różni się od deepchecks, giskard, evidently i fairlearn. Do E6.
- **O-5:** Wersjonowanie danych (DVC, Git LFS albo poza repo). Do oddania pracy.
- **O-7:** Czy filtr korelacyjny wchodzi do Pipeline. Kosztuje 0,15 R² na modelu liniowym, zysk może być w stabilności SHAP. Rozstrzygnę w E9.
- **O-8:** M3 robi early stopping na zbiorze kalibracyjnym, a ten sam zbiór ma kalibrować przedziały w W3. Kalibracja przestaje być wtedy całkiem "czysta". Do rozwiązania w E8, np. early stopping na wydzielonej części treningu.
