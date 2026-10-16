# trustml

**A framework for evaluating how far a machine learning model can be trusted, not just how accurate it is.**

MSc thesis project, AGH University of Krakow (Computer Science – Data Science, defence planned for July 2027).
Status: work in progress. The data pipeline, validation protocol, baselines and first models are done; the remaining reliability dimensions are built stage by stage (see [Roadmap](#roadmap)).

## The idea

A single accuracy score says little about how a model behaves in practice. `trustml` takes any scikit-learn–compatible model and produces a standardised report across six dimensions:

| | Dimension | Question it answers | Main measure |
|---|---|---|---|
| W1 | Predictive accuracy | How close are the predictions? | RMSLE, MdAPE |
| W2 | Robustness | How much worse does it get when conditions change? | Degradation under domain shift and noise, split into bias and variance |
| W3 | Uncertainty | Does it know what it doesn't know? | Coverage and width of conformal prediction intervals |
| W4 | Explainability | Are its explanations stable? | SHAP importance + Kendall's tau across seeds and folds |
| W5 | Fairness | Is it equally accurate and unbiased across groups? | Per-group error gaps and mean residual (region, age, position, league) |
| W6 | Reproducibility | Can the result be recreated? | Metric spread across random seeds, complete run records |

Working hypothesis: **model rankings differ between dimensions.** The most accurate model is not necessarily the most robust or the best calibrated. The main result will be a model × dimension matrix with rank correlations between dimensions.

## Case study: football player market values

The demonstration task is predicting a player's Transfermarkt market value from match statistics (FBref) in the Big 5 European leagues.

- **Dataset:** 13,709 player-seasons × 115 columns, 5,310 players, seasons 2017/18–2023/24 (2019/20 excluded: Ligue 1 played only 279 of 380 matches).
- **Target:** `log1p(market value in EUR)`.
- **Temporal split:** train 2017/18–2021/22, calibration 2022/23, test 2023/24. A random split would put the same player in train and test, and market values are strongly correlated year to year.
- **Domain shift set:** the Portuguese Primeira Liga is held out entirely, to measure how the model behaves on a league it has never seen.
- **Hyperparameter tuning:** Optuna with GroupKFold by player inside the training set only; the test set stays untouched.

### Results so far (stage E5, test set 2023/24)

| Model | RMSLE ↓ | R² (log scale) ↑ |
|---|---|---|
| M0a – global median | 1.27 | −0.05 |
| M0b – median per position × league | 1.17 | 0.10 |
| M0c – last season's value, full test (missing filled with M0b) | 0.76 | 0.62 |
| M1 – Elastic Net | 0.67 | 0.71 |
| M2 – Random Forest | 0.70 | 0.68 |
| M3 – XGBoost | 0.65 | 0.73 |
| M0c – last season's value, only players with a value last season (72% of test) | 0.53 | 0.81 |

Notes:

- **Last season's value is a reference point, not a competitor.** Models are not given the previous valuation, so they have to work from on-pitch statistics. On the players who have a previous value, simply copying it is better than any model. On the full test set, where 28% of players have no previous value, all three models beat it. Valuations are sticky from year to year, which matches the conclusion of my bachelor's thesis on the same problem.
- **Overall numbers hide differences between groups.** Every metric is also reported per league, position and "player seen in training vs. new player". For example, M3 has R² 0.76 on known players and 0.65 on new ones.

![Age vs. market value in the Big 5 leagues](reports/figures/eda/a3_krzywa_wieku.png)

*From the exploratory analysis: the fitted curve peaks around age 23–24, and forwards are valued highest at almost every age. The very young look expensive because the 225-minute filter keeps only those already playing in the Big 5. Figure labels are in Polish.*

## Engineering choices

- **Leakage tests:** `tests/test_no_leakage.py` checks that no player-season appears in two sets and that the Primeira Liga never reaches training or calibration.
- **Fixed evaluation protocol:** `configs/split.yaml` fixes seasons, grouping, metrics and the SHA-256 of the dataset it applies to, so the rules cannot quietly change after seeing results.
- **Data provenance:** `data/processed/manifest.json` stores checksums of every input file, library versions, all parameters and row counts after each pipeline step.
- **Validated merges:** every join uses `validate=`. The previous version of this pipeline joined players by name without validation and silently produced 846 duplicated player-season pairs.
- **Decision log:** every non-obvious choice (metrics, deflation, back-transformation from log scale, etc.) is recorded with its reason in `docs/decyzje.md`.

## Project structure

```
src/trustml/            the library, domain-independent (metrics, splits, baselines, pipeline)
case_study/football/    data pipeline, EDA and modelling for the case study
configs/                evaluation protocol
docs/                   data, decision log, plan (Polish)
reports/                figures and result tables
tests/                  data, metric, pipeline and leakage tests
```

## Getting started

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest     # data-dependent tests are skipped if the dataset is absent
```

Raw data is not included. FBref match data (`data/raw/fbref/master.db`) and Transfermarkt valuations from the Kaggle dataset [`davidcariboo/player-scores`](https://www.kaggle.com/datasets/davidcariboo/player-scores) (`players.csv`, `player_valuations.csv` in `data/raw/tm/`) are needed to rebuild the dataset:

```bash
python -m case_study.football.przygotuj_dane       # build the modelling dataset
python -m case_study.football.uruchom_eda          # exploratory analysis
python -m case_study.football.poziom_odniesienia   # baselines M0a–M0c
python -m case_study.football.modelowanie          # tune and evaluate M1–M3
```

## Roadmap

| Stage | Scope | Status |
|---|---|---|
| E1 | Data pipeline | ✅ |
| E2 | Provenance manifest, tests, imports | ✅ |
| E3 | Exploratory data analysis | ✅ |
| E4 | Validation protocol and baselines | ✅ |
| E5 | Models M1–M3, accuracy (W1) | in progress |
| E6 | `trustml` 0.1: library skeleton, W1, W6 | planned |
| E7 | Robustness (W2) | planned |
| E8 | Uncertainty (W3) | planned |
| E9 | Explainability (W4) | planned |
| E10 | Fairness (W5) | planned |
| E11 | Domain shift: Primeira Liga | planned |
| E12 | `trustml` 1.0 and synthesis | planned |

## Stack

Python 3.12, pandas, NumPy, scikit-learn, XGBoost, Optuna, SciPy, Matplotlib, PyArrow, pytest, ruff, pre-commit, uv.

## Author

Wojciech Niemiec · [LinkedIn](https://www.linkedin.com/in/wojciechniemiec) · supervised at AGH University of Krakow
