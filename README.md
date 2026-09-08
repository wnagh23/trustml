# trustml

Framework oceny wiarygodności modeli uczenia maszynowego.

Biblioteka przyjmuje dowolny model zgodny z API scikit-learn i produkuje
wystandaryzowany raport w sześciu wymiarach: poprawność predykcyjna (W1), odporność
(W2), niepewność (W3), wyjaśnialność (W4), sprawiedliwość (W5) i odtwarzalność (W6).

## Struktura

- `src/trustml/` — biblioteka, niezależna od dziedziny
- `configs/` — protokół oceny zamrożony w plikach konfiguracyjnych
- `models/` — wytrenowane modele, poza gitem
- `case_study/football/` — pipeline danych dziedzinowych
- `data/raw/` — źródła, tylko do odczytu, poza gitem (około 2,4 GB)
- `data/processed/` — zbiór modelowy
- `docs/` — dokumentacja
- `reports/` — figury i tabele wynikowe
- `tests/` — testy

## Środowisko

```
uv sync
uv pip install -e .
```

Python 3.12 lub nowszy. Zależności w `pyproject.toml`, zamrożone w `uv.lock`.

## Odtworzenie zbioru modelowego

Wymaga `data/raw/fbref/master.db` (221 MB) oraz `data/raw/tm/players.csv`
i `data/raw/tm/player_valuations.csv` ze zbioru Kaggle `davidcariboo/player-scores`.
Dane leżą poza repozytorium.

```
python -m case_study.football.przygotuj_dane
```

Wynik trafia do `data/processed/`:
`zawodnik_sezon.parquet` jako pełny zbiór kontrolny, `model.parquet` jako zbiór
modelowy o wymiarach 13 709 na 114, oraz `manifest.json` z prowieniencją — sumami
kontrolnymi, wersjami i parametrami.

Parametry przebiegu, czyli progi, sezony i ścieżki, siedzą
w `case_study/football/parametry.py`.

## Analiza eksploracyjna

```
python -m case_study.football.uruchom_eda
```

Przebieg trwa około dziesięciu sekund i czyta wyłącznie `data/processed/`. Zapisuje
figury do `reports/figures/eda/`, tabele do `reports/tables/eda/` oraz
`reports/tables/eda/fakty.json` z kompletem zmierzonych liczb. Omówienie wyników
znajduje się w `docs/05-eda.md`.

## Protokół walidacji i poziom odniesienia

```
python -m case_study.football.poziom_odniesienia
```

Przebieg trwa kilka sekund. Liczy trzy poziomy odniesienia — medianę globalną,
medianę w komórce pozycja × liga oraz przepisanie wyceny z poprzedniego sezonu —
na zbiorze kalibracyjnym i testowym, z podziałem na osie z D-12. Zapisuje
`reports/tables/e4_baseline.csv` oraz dwie tabele diagnostyczne opisujące sam
podział: `e4_przeciecia.csv` i `e4_foldy.csv`.

Zasady oceny są zamrożone w `configs/split.yaml`: sezony w każdym zbiorze, kolumna
grupująca dla GroupKFold, lista metryk, osie stratyfikacji i suma kontrolna pliku
`model.parquet`, którego dotyczą. Po każdym przebiegu pipeline'u zmieniającym zbiór
modelowy trzeba wpisać tam nową sumę i przeliczyć wyniki — do tego czasu
`tests/test_no_leakage.py` nie przechodzi.

## Modele i wymiar W1

```
python -m case_study.football.modelowanie      # strojenie, uczenie, ocena
python -m case_study.football.warianty         # analizy wrażliwości
```

Pierwsza komenda stroi trzy modele — Elastic Net, Random Forest i XGBoost —
setką prób optuny na zbiorze treningowym, przez GroupKFold po zawodniku.
Przebieg trwa kilkadziesiąt minut; liczbę prób można podać argumentem, na przykład
`... modelowanie 5` przy diagnozowaniu. Zapisuje modele do `models/`, metryki do
`reports/tables/e5_w1.csv`, dziennik prób do `e5_przebiegi.csv`, analizę reszt do
`e5_reszty.csv` i metryczkę przebiegu do `e5_metryczka.json`.

Druga komenda bierze najlepszy model z metryczki i liczy trzy warianty: cel
zdeflowany zamiast nominalnego (D-04), zbiór cech z przywróconą parą wskaźników
dryblingu (D-25) oraz filtr korelacyjny przy progu 0,95 (O-7). Wynik trafia do
`reports/tables/e5_warianty.csv`.

Metryki liczy ten sam kod, który policzył poziomy odniesienia w E4, więc wiersze
z `e5_w1.csv` i `e4_baseline.csv` porównuje się wprost.

## Testy

```
pytest
ruff check
ruff format
```

## Dokumentacja

- `docs/01-projekt.md` — cel pracy, wymiary W1–W6, modele, API biblioteki
- `docs/02-dane.md` — źródła, pipeline, słownik zbioru, ograniczenia
- `docs/03-decyzje.md` — log decyzji projektowych
- `docs/04-plan.md` — mapa drogowa i status
- `docs/05-eda.md` — analiza eksploracyjna, wnioski i ciekawostki

## Źródła danych

FBref dostarcza statystyki mecz po meczu, a Transfermarkt — przez zbiór Kaggle
`davidcariboo/player-scores` — wyceny i cechy statyczne.
