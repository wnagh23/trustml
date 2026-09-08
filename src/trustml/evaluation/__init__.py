"""
Protokol oceny: podzial danych, metryki i poziomy odniesienia.

Moduly tego pakietu nie wiedza nic o pilce noznej. Nazwy kolumn wchodza
parametrami, zeby ten sam kod obsluzyl przejscie na inny zbior danych - test
takiego przejscia jest zaplanowany w E12 (D-19).
"""

from .baselines import (
    mediana_globalna,
    mediana_komorki,
    przepisanie_z_poprzedniego,
    uzupelnij,
)
from .metrics import (
    KOLUMNY_METRYK,
    do_euro,
    krotnosc_bledu,
    mae_euro,
    mape,
    mdape,
    metryki,
    ocen,
    r2_log,
    rmsle,
    skala_agregatu,
    wspolczynnik_duana,
)
from .splits import (
    pary,
    podsumowanie_foldow,
    podzial_grupowy,
    policz_sha256,
    raport_przeciec,
    sprawdz_odcisk_danych,
    wczytaj_konfiguracje,
)

__all__ = [
    "KOLUMNY_METRYK",
    "do_euro",
    "krotnosc_bledu",
    "mae_euro",
    "mape",
    "mdape",
    "mediana_globalna",
    "mediana_komorki",
    "metryki",
    "ocen",
    "pary",
    "podsumowanie_foldow",
    "podzial_grupowy",
    "policz_sha256",
    "przepisanie_z_poprzedniego",
    "r2_log",
    "raport_przeciec",
    "rmsle",
    "skala_agregatu",
    "sprawdz_odcisk_danych",
    "uzupelnij",
    "wczytaj_konfiguracje",
    "wspolczynnik_duana",
]
