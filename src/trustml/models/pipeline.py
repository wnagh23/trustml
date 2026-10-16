"""
Wspolne przetwarzanie cech dla wszystkich modeli: imputacja z flaga braku (D-26),
standaryzacja, opcjonalny filtr korelacyjny (O-7) i one-hot bez drop (D-10).
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
    Usuwa jedna ceche z kazdej pary o |korelacji| powyzej `prog`.

    Z pary zostaje cecha wczesniejsza, wiec wynik jest powtarzalny.
    `prog=None` wylacza filtr.
    """

    def __init__(self, prog: float | None = 0.95):
        self.prog = prog

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_cech_wejsciowych_ = X.shape[1]

        if self.prog is None:
            self.zachowane_ = np.ones(X.shape[1], dtype=bool)
            return self

        # kolumna stala daje NaN w macierzy korelacji, traktuje ja jak nieskorelowana
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
        """Zwraca nazwy cech, ktore przeszly przez filtr."""
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
    """Sklada ColumnTransformer dla kolumn liczbowych i kategorycznych."""
    liczbowe = Pipeline(
        [
            # add_indicator dokleja kolumne 0/1 "byl brak" dla cech z brakami
            ("imputacja", SimpleImputer(strategy="median", add_indicator=True)),
            ("skalowanie", StandardScaler()),
            ("filtr", FiltrKorelacyjny(prog=prog_korelacji)),
        ]
    )
    # bez drop: nieznana liga (Primeira Liga) daje same zera, a nie udaje
    # kategorii referencyjnej
    kategoryczne = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        [
            ("liczbowe", liczbowe, list(kolumny_liczbowe)),
            ("kategoryczne", kategoryczne, list(kolumny_kategoryczne)),
        ],
        # do modelu wchodzi tylko to, co wymienione; w pliku sa tez cel i klucze
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
    """Zwraca Pipeline: wspolne przetwarzanie + podany estymator."""
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
