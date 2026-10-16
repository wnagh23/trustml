"""
Etap E5: strojenie, uczenie i ocena modeli M1-M3 (wymiar W1).

    python -m case_study.football.modelowanie [liczba_prob]

Strojenie: optuna + GroupKFold po zawodniku, tylko na treningu.
Early stopping M3 na kalibracji, test oceniany na koncu.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "case_study.football"

from trustml.evaluation import (
    ocen,
    podzial_grupowy,
    policz_sha256,
    rmsle,
    sprawdz_odcisk_danych,
    wczytaj_konfiguracje,
    wspolczynnik_duana,
)
from trustml.models import zbuduj_model

from .parametry import KATALOG_REPO, KATALOG_WYJSCIA

SCIEZKA_MODELU = KATALOG_WYJSCIA / "model.parquet"
SCIEZKA_KONFIGURACJI = KATALOG_REPO / "configs" / "split.yaml"
KATALOG_TABEL = KATALOG_REPO / "reports" / "tables"
KATALOG_MODELI = KATALOG_REPO / "models"

ZIARNO = 20260829

# Kolumny, ktore nie moga wejsc do modelu: identyfikatory, podzial i cele.
KOLUMNY_META = [
    "player_id",
    "season",
    "name",
    "tm_player_id",
    "podzial",
    "znany_z_treningu",
    "market_value_in_eur",
    "market_value_real",
    "log_market_value",
    "log_market_value_real",
]

KOLUMNY_KATEGORYCZNE = ["pozycja", "region", "liga", "noga", "zmienil_lige"]

# Para dryblingu ze zmiana definicji na granicy trening/kalibracja (D-25).
CECHY_ZMIENIONE = ["dribble_success_percentage", "tackled_perecentage"]

# Skok na tej samej granicy, ale bez dowodu zmiany definicji; zostaje (D-25).
CECHA_GRANICZNA = "successful_dribbler_tackle_percentage"

PROB_STROJENIA = 100


def kolumny_cech(dane: pd.DataFrame, *, bez_zmienionych: bool = True) -> tuple:
    """Zwraca (kolumny liczbowe, kolumny kategoryczne) uzywane jako cechy."""
    odrzucone = set(KOLUMNY_META) | set(KOLUMNY_KATEGORYCZNE)
    if bez_zmienionych:
        odrzucone |= set(CECHY_ZMIENIONE)
    liczbowe = [k for k in dane.columns if k not in odrzucone]
    return liczbowe, list(KOLUMNY_KATEGORYCZNE)


def wczytaj() -> tuple[pd.DataFrame, dict]:
    """Wczytuje zbior modelowy i split.yaml, sprawdzajac odcisk danych."""
    konfiguracja = wczytaj_konfiguracje(SCIEZKA_KONFIGURACJI)
    sprawdz_odcisk_danych(konfiguracja, SCIEZKA_MODELU)
    return pd.read_parquet(SCIEZKA_MODELU), konfiguracja


def _wynik_walidacji(
    model, X, y, grupy, n_podzialow: int, proba: optuna.Trial | None = None
) -> float:
    """
    Sredni RMSLE po foldach GroupKFold.

    Po kazdym foldzie zglasza wynik do optuny, zeby slabe proby mozna bylo przerwac.
    """
    wyniki = []
    for numer, (ucz, wal) in enumerate(podzial_grupowy(grupy, n_podzialow)):
        model.fit(X.iloc[ucz], y[ucz])
        wyniki.append(rmsle(y[wal], model.predict(X.iloc[wal])))
        if proba is not None:
            proba.report(float(np.mean(wyniki)), numer)
            if proba.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(wyniki))


def przestrzenie(proba: optuna.Trial, nazwa: str) -> dict:
    """Przestrzen hiperparametrow dla modelu "M1", "M2" albo "M3"."""
    if nazwa == "M1":
        return {
            "alpha": proba.suggest_float("alpha", 1e-4, 1e1, log=True),
            "l1_ratio": proba.suggest_float("l1_ratio", 0.0, 1.0),
        }
    if nazwa == "M2":
        return {
            "n_estimators": proba.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": proba.suggest_int("max_depth", 4, 24),
            "min_samples_leaf": proba.suggest_int("min_samples_leaf", 1, 20),
            "max_features": proba.suggest_float("max_features", 0.1, 1.0),
        }
    if nazwa == "M3":
        return {
            "n_estimators": proba.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate": proba.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": proba.suggest_int("max_depth", 2, 10),
            "subsample": proba.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": proba.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": proba.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": proba.suggest_float("reg_lambda", 1e-3, 1e2, log=True),
        }
    raise ValueError(f"nieznany model: {nazwa}")


def zbuduj_estymator(nazwa: str, parametry: dict):
    """Tworzy estymator M1, M2 albo M3 z podanymi hiperparametrami."""
    if nazwa == "M1":
        # przy malej alfie domyslne max_iter=1000 nie wystarcza do zbieznosci
        return ElasticNet(max_iter=20_000, random_state=ZIARNO, **parametry)
    if nazwa == "M2":
        return RandomForestRegressor(random_state=ZIARNO, n_jobs=-1, **parametry)
    if nazwa == "M3":
        return XGBRegressor(
            random_state=ZIARNO,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
            **parametry,
        )
    raise ValueError(f"nieznany model: {nazwa}")


def stroj(
    nazwa: str,
    X: pd.DataFrame,
    y: np.ndarray,
    grupy: np.ndarray,
    kolumny: tuple,
    n_podzialow: int,
    n_prob: int,
) -> tuple[dict, pd.DataFrame]:
    """Stroi model optuna na treningu; zwraca (najlepsze parametry, dziennik prob)."""
    liczbowe, kategoryczne = kolumny

    def cel(proba: optuna.Trial) -> float:
        parametry = przestrzenie(proba, nazwa)
        model = zbuduj_model(zbuduj_estymator(nazwa, parametry), liczbowe, kategoryczne)
        return _wynik_walidacji(model, X, y, grupy, n_podzialow)

    badanie = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=ZIARNO),
        # pierwsze 10 prob bez przycinania, zeby bylo z czego liczyc mediane
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        study_name=nazwa,
    )
    start = time.perf_counter()
    badanie.optimize(cel, n_trials=n_prob, show_progress_bar=False)
    czas = time.perf_counter() - start

    dziennik = badanie.trials_dataframe(attrs=("number", "value", "params", "state"))
    dziennik.insert(0, "model", nazwa)
    przyciete = int((dziennik["state"] == "PRUNED").sum())
    print(
        f"    {nazwa}: {n_prob} prob w {czas / 60:.1f} min "
        f"({przyciete} przycietych), najlepszy RMSLE {badanie.best_value:.4f}"
    )
    return badanie.best_params, dziennik


def dopasuj_finalny(
    nazwa: str,
    parametry: dict,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_kal: pd.DataFrame,
    y_kal: np.ndarray,
    kolumny: tuple,
):
    """Uczy model na calym treningu; M3 z early stoppingiem na kalibracji."""
    liczbowe, kategoryczne = kolumny
    if nazwa == "M3":
        estymator = zbuduj_estymator(
            nazwa, {**parametry, "early_stopping_rounds": 50, "eval_metric": "rmse"}
        )
        model = zbuduj_model(estymator, liczbowe, kategoryczne)
        # eval_set musi przejsc przez to samo przetwarzanie co trening
        przetwarzanie = model.named_steps["przetwarzanie"]
        X_tr_p = przetwarzanie.fit_transform(X_tr, y_tr)
        X_kal_p = przetwarzanie.transform(X_kal)
        model.named_steps["model"].fit(
            X_tr_p, y_tr, eval_set=[(X_kal_p, y_kal)], verbose=False
        )
        return model

    model = zbuduj_model(zbuduj_estymator(nazwa, parametry), liczbowe, kategoryczne)
    model.fit(X_tr, y_tr)
    return model


def ocen_model(
    model,
    nazwa: str,
    dane: pd.DataFrame,
    X_all: pd.DataFrame,
    duan: float,
    konfiguracja: dict,
    zbiory: tuple[str, ...],
) -> pd.DataFrame:
    """Metryki modelu na podanych zbiorach z podzialem na osie z split.yaml."""
    cel = konfiguracja["cel"]["glowny"]
    osie = konfiguracja["osie_stratyfikacji"]

    czesci = []
    for zbior in zbiory:
        maska = (dane["podzial"] == zbior).to_numpy()
        podzbior = dane[maska]
        wynik = ocen(
            podzbior[cel].to_numpy(),
            model.predict(X_all[maska]),
            {os: podzbior[os].to_numpy() for os in osie},
            duan=duan,
            etykieta=nazwa,
        )
        wynik.insert(0, "zbior", zbior)
        czesci.append(wynik)
    return pd.concat(czesci, ignore_index=True)


def analiza_reszt(
    reszty: np.ndarray, podzbior: pd.DataFrame, predykcja: np.ndarray, nazwa: str
) -> pd.DataFrame:
    """
    Srednia i odchylenie reszt w decylach predykcji i prawdy, pozycjach i ligach.

    Reszta dodatnia = model zanizyl.
    """
    wiersze = []

    # qcut dzieli na grupy o rownej liczebnosci
    for etykieta, os in [
        ("decyl_predykcji", pd.qcut(predykcja, 10, labels=False, duplicates="drop")),
        (
            "decyl_prawdy",
            pd.qcut(podzbior["log_market_value"], 10, labels=False, duplicates="drop"),
        ),
        ("pozycja", podzbior["pozycja"].to_numpy()),
        ("liga", podzbior["liga"].to_numpy()),
    ]:
        os = np.asarray(os)
        for wartosc in sorted(pd.unique(os), key=str):
            w = os == wartosc
            wiersze.append(
                {
                    "model": nazwa,
                    "analiza": etykieta,
                    "grupa": str(wartosc),
                    "n": int(w.sum()),
                    "srednia_reszty": float(reszty[w].mean()),
                    "odchylenie_reszty": float(reszty[w].std()),
                    "srednia_predykcji": float(predykcja[w].mean()),
                }
            )
    return pd.DataFrame(wiersze)


def main(n_prob: int = PROB_STROJENIA) -> None:
    """Stroi, uczy i ocenia M1-M3; zapisuje tabele e5_* do reports/tables/."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    KATALOG_TABEL.mkdir(parents=True, exist_ok=True)
    KATALOG_MODELI.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"MODELE M1-M3 I WYMIAR W1  (etap E5, {n_prob} prob na model)")
    print("=" * 78)

    dane, konfiguracja = wczytaj()
    cel = konfiguracja["cel"]["glowny"]
    grupa = konfiguracja["podzial_wewnetrzny"]["kolumna_grupujaca"]
    n_podzialow = konfiguracja["podzial_wewnetrzny"]["n_podzialow"]

    kolumny = kolumny_cech(dane)
    liczbowe, kategoryczne = kolumny
    print(
        f"\n  cechy: {len(liczbowe)} liczbowych + {len(kategoryczne)} kategorycznych"
        f"  (odrzucone przez O-6: {', '.join(CECHY_ZMIENIONE)})"
    )

    X_all = dane[liczbowe + kategoryczne]
    trening = dane["podzial"] == "trening"
    kalibracja = dane["podzial"] == "kalibracja"

    X_tr, y_tr = X_all[trening.to_numpy()], dane.loc[trening, cel].to_numpy()
    X_kal, y_kal = X_all[kalibracja.to_numpy()], dane.loc[kalibracja, cel].to_numpy()
    grupy_tr = dane.loc[trening, grupa].to_numpy()

    wyniki, dzienniki, reszty_wszystkie, najlepsze = [], [], [], {}

    for nazwa in ["M1", "M2", "M3"]:
        print(f"\n  --- {nazwa} ---")
        parametry, dziennik = stroj(
            nazwa, X_tr, y_tr, grupy_tr, kolumny, n_podzialow, n_prob
        )
        dzienniki.append(dziennik)
        najlepsze[nazwa] = parametry

        model = dopasuj_finalny(nazwa, parametry, X_tr, y_tr, X_kal, y_kal, kolumny)
        joblib.dump(model, KATALOG_MODELI / f"{nazwa.lower()}.joblib")

        # wspolczynnik Duana tylko z reszt treningowych (D-21)
        reszty_tr = y_tr - model.predict(X_tr)
        duan = wspolczynnik_duana(reszty_tr)

        wynik = ocen_model(
            model, nazwa, dane, X_all, duan, konfiguracja, ("kalibracja", "test")
        )
        wyniki.append(wynik)

        for zbior in ["kalibracja", "test"]:
            w = wynik[(wynik["zbior"] == zbior) & (wynik["os"] == "calosc")].iloc[0]
            print(
                f"    {zbior:11s} RMSLE {w['rmsle']:.4f}"
                f"  krotnosc {w['krotnosc']:.2f}"
                f"  MdAPE {w['mdape']:5.1f}%"
                f"  R2 {w['r2_log']:6.4f}"
                f"  MAE {w['mae_eur'] / 1e6:5.2f} mln"
                f"  agregat {w['agregat']:.2f} -> {w['agregat_po_korekcie']:.2f}"
            )

        maska_kal = kalibracja.to_numpy()
        predykcja_kal = model.predict(X_all[maska_kal])
        reszty_wszystkie.append(
            analiza_reszt(y_kal - predykcja_kal, dane[maska_kal], predykcja_kal, nazwa)
        )

    pd.concat(wyniki, ignore_index=True).to_csv(
        KATALOG_TABEL / "e5_w1.csv", index=False
    )
    pd.concat(dzienniki, ignore_index=True).to_csv(
        KATALOG_TABEL / "e5_przebiegi.csv", index=False
    )
    pd.concat(reszty_wszystkie, ignore_index=True).to_csv(
        KATALOG_TABEL / "e5_reszty.csv", index=False
    )

    metryczka = {
        "ziarno": ZIARNO,
        "prob_strojenia": n_prob,
        "sha256_danych": policz_sha256(SCIEZKA_MODELU),
        "sha256_konfiguracji": policz_sha256(SCIEZKA_KONFIGURACJI),
        "cel": cel,
        "cechy_liczbowe": len(liczbowe),
        "cechy_kategoryczne": len(kategoryczne),
        "cechy_odrzucone_o6": CECHY_ZMIENIONE,
        "najlepsze_hiperparametry": najlepsze,
    }
    (KATALOG_TABEL / "e5_metryczka.json").write_text(
        json.dumps(metryczka, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  zapisano wyniki do {KATALOG_TABEL.relative_to(KATALOG_REPO)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else PROB_STROJENIA)
