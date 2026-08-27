# trustml

Framework oceny wiarygodności modeli uczenia maszynowego.

Biblioteka przyjmuje dowolny model zgodny z API scikit-learn i produkuje
wystandaryzowany raport w sześciu wymiarach: poprawność predykcyjna (W1), odporność
(W2), niepewność (W3), wyjaśnialność (W4), sprawiedliwość (W5) i odtwarzalność (W6).

## Struktura

- `src/trustml/` — biblioteka, niezależna od dziedziny
- `case_study/football/` — pipeline danych dziedzinowych
- `data/raw/` — źródła, tylko do odczytu, poza gitem (około 2,4 GB)
- `data/processed/` — zbiór modelowy
- `docs/` — dokumentacja
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

## Źródła danych

FBref dostarcza statystyki mecz po meczu, a Transfermarkt — przez zbiór Kaggle
`davidcariboo/player-scores` — wyceny i cechy statyczne.
