"""
Wspolny Pipeline przetwarzania cech dla wszystkich modeli.

Kazdy model dostaje dokladnie to samo wejscie, wiec roznice w wynikach pochodza
z samego algorytmu. Kolejnosc krokow:

    1. imputacja mediana, z flaga braku (add_indicator)
    2. standaryzacja
    3. opcjonalny filtr korelacyjny
    4. kodowanie kategorii bez upuszczania poziomu

Punkt pierwszy realizuje O-3. Braki w kolumnach procentowych sa strukturalne,
czyli oznaczaja mianownik rowny zeru, i rozkladaja sie nierowno miedzy pozycjami
- 4,88 procent u obroncow wobec 0,17 u napastnikow. Sama imputacja mediana
wprowadzilaby obciazenie grupowe, wiec obok wartosci wstawionej idzie kolumna
zero-jedynkowa mowiaca, ze wartosci tam nie bylo. Model moze ja wykorzystac,
a W5 ma czego szukac.

Punkt czwarty realizuje D-10. `OneHotEncoder` bez `drop` i z
`handle_unknown="ignore"`: upuszczenie pierwszego poziomu rozklada wage kategorii
odniesienia na wyraz wolny, przez co ranking waznosci cech w W4 przestaje byc
porownywalny miedzy modelami. Przy nieznanej kategorii - a taka bedzie Primeira
Liga w E11 - koder zwraca same zera zamiast podnosic wyjatek.

Filtr korelacyjny jest opcjonalny i domyslnie wylaczony, bo jego bilans jest
przedmiotem O-7: przy progu 0,95 kosztuje 0,15 R2 na modelu liniowym, a zysk lezy
w stabilnosci rankingu SHAP, czyli po stronie W4. Obie polowy bilansu mierzymy
na tej samej konfiguracji.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


class FiltrKorelacyjny(BaseEstimator, TransformerMixin):
    """
    Usuwa jedna ceche z kazdej pary skorelowanej powyzej progu.

    Kolejnosc przegladania jest ustalona i idzie od lewej: z pary zostaje cecha
    wczesniejsza w kolumnach wejsciowych. Determinizm ma tu znaczenie praktyczne,
    bo inaczej ranking waznosci cech zmienialby sie miedzy przebiegami
    z powodow niezwiazanych z modelem (W6).

    Przyjmuje:
        prog - wartosc bezwzgledna korelacji, powyzej ktorej para uchodzi
               za redundantna; None wylacza filtr
    """

    def __init__(self, prog: float | None = 0.95):
        self.prog = prog

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_cech_wejsciowych_ = X.shape[1]

        if self.prog is None:
            self.zachowane_ = np.ones(X.shape[1], dtype=bool)
            return self

        # rowvar=False mowi numpy, ze kolumny sa zmiennymi, a wiersze
        # obserwacjami. Kolumna stala daje dzielenie przez zero i NaN
        # w macierzy - taka pare traktujemy jako nieskorelowana.
        with np.errstate(invalid="ignore", divide="ignore"):
            korelacje = np.abs(np.corrcoef(X, rowvar=False))
        korelacje = np.nan_to_num(korelacje, nan=0.0)

        zachowane = np.ones(X.shape[1], dtype=bool)
        for i in range(X.shape[1]):
            if not zachowane[i]:
                continue
            for j in range(i + 1, X.shape[1]):
                if zachowane[j] and korelacje[i, j] > self.prog:
                    zachowane[j] = False
        self.zachowane_ = zachowane
        return self

    def transform(self, X):
        check_is_fitted(self, "zachowane_")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_cech_wejsciowych_:
            raise ValueError(
                f"filtr uczyl sie na {self.n_cech_wejsciowych_} cechach, "
                f"dostal {X.shape[1]}"
            )
        return X[:, self.zachowane_]

    def get_feature_names_out(self, input_features=None):
        """Przekazuje dalej nazwy cech, ktore przezyly filtr - potrzebne w W4."""
        check_is_fitted(self, "zachowane_")
        if input_features is None:
            input_features = np.array(
                [f"x{i}" for i in range(self.n_cech_wejsciowych_)]
            )
        return np.asarray(input_features)[self.zachowane_]


def zbuduj_przetwarzanie(
    kolumny_liczbowe: Sequence[str],
    kolumny_kategoryczne: Sequence[str],
    *,
    prog_korelacji: float | None = None,
) -> ColumnTransformer:
    """
    Sklada przetwarzanie cech wspolne dla wszystkich modeli.

    Przyjmuje:
        kolumny_liczbowe     - nazwy kolumn ciaglych
        kolumny_kategoryczne - nazwy kolumn tekstowych
        prog_korelacji       - prog filtra korelacyjnego; None wylacza filtr

    Zwraca:
        ColumnTransformer gotowy do wstawienia do Pipeline.
    """
    liczbowe = Pipeline(
        [
            # add_indicator dokleja kolumne zero-jedynkowa dla kazdej cechy
            # z brakami; to jest wariant "flaga zero prob" z O-3.
            ("imputacja", SimpleImputer(strategy="median", add_indicator=True)),
            ("skalowanie", StandardScaler()),
            ("filtr", FiltrKorelacyjny(prog=prog_korelacji)),
        ]
    )
    kategoryczne = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        [
            ("liczbowe", liczbowe, list(kolumny_liczbowe)),
            ("kategoryczne", kategoryczne, list(kolumny_kategoryczne)),
        ],
        # remainder="drop": kolumna, ktorej nikt nie wymienil, ma nie wejsc
        # do modelu przez przypadek. Klucze i metadane leza w tym samym pliku.
        remainder="drop",
        verbose_feature_names_out=False,
    )


def zbuduj_model(
    estymator,
    kolumny_liczbowe: Sequence[str],
    kolumny_kategoryczne: Sequence[str],
    *,
    prog_korelacji: float | None = None,
) -> Pipeline:
    """
    Doklada estymator na koniec wspolnego przetwarzania.

    Przyjmuje:
        estymator            - dowolny regresor zgodny z API scikit-learn
        kolumny_liczbowe     - nazwy kolumn ciaglych
        kolumny_kategoryczne - nazwy kolumn tekstowych
        prog_korelacji       - prog filtra korelacyjnego; None wylacza filtr

    Zwraca:
        Pipeline uczacy sie od surowej ramki do predykcji.
    """
    return Pipeline(
        [
            (
                "przetwarzanie",
                zbuduj_przetwarzanie(
                    kolumny_liczbowe,
                    kolumny_kategoryczne,
                    prog_korelacji=prog_korelacji,
                ),
            ),
            ("model", estymator),
        ]
    )
