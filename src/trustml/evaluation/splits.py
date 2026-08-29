"""
Protokol podzialu danych - zamrozona konfiguracja i walidacja krzyzowa.

Sa tu dwa poziomy podzialu i mylenie ich jest najczestszym zrodlem falszywie
dobrych wynikow.

Podzial zewnetrzny jest czasowy i mieszka w danych, w kolumnie "podzial"
(trening, kalibracja, test, zbior_C). Ten modul go nie tworzy - odczytuje jego
deklaracje z `configs/split.yaml` i sprawdza, czy plik danych ma odcisk zapisany
w konfiguracji. Bez tego sprawdzenia porownanie wynikow z dwoch dni moze
niepostrzezenie dotyczyc dwoch roznych zbiorow.

Podzial wewnetrzny sluzy wylacznie strojeniu hiperparametrow i dzieli sam zbior
treningowy. Uzywamy GroupKFold po zawodniku: ten sam zawodnik wystepuje
w treningu w czterech sezonach, a jego wycena jest z roku na rok mocno
skorelowana, wiec zwykly KFold rozrzucilby jego sezony po czesci uczacej
i walidacyjnej i pokazal wynik lepszy od prawdziwego.

GroupKFold jest deterministyczny przy domyslnych ustawieniach, co jest wymogiem
wymiaru W6.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold


def policz_sha256(sciezka: Path, kawalek: int = 1024 * 1024) -> str:
    """
    Liczy odcisk SHA-256 pliku, czytajac go porcjami.

    Przyjmuje:
        sciezka - plik do zahaszowania
        kawalek - rozmiar porcji w bajtach

    Zwraca:
        odcisk jako 64 znaki szesnastkowe.
    """
    skrot = hashlib.sha256()
    with Path(sciezka).open("rb") as plik:
        while porcja := plik.read(kawalek):
            skrot.update(porcja)
    return skrot.hexdigest()


def wczytaj_konfiguracje(sciezka: Path) -> dict[str, Any]:
    """
    Wczytuje `split.yaml` do slownika.

    Przyjmuje:
        sciezka - plik konfiguracji

    Zwraca:
        slownik odwzorowujacy strukture pliku.
    """
    sciezka = Path(sciezka)
    if not sciezka.exists():
        raise FileNotFoundError(f"brak konfiguracji podzialu: {sciezka}")
    # safe_load buduje wylacznie zwykle typy Pythona; pelny load potrafi
    # skonstruowac dowolny obiekt zapisany w pliku.
    return yaml.safe_load(sciezka.read_text(encoding="utf-8"))


def sprawdz_odcisk_danych(
    konfiguracja: dict[str, Any],
    sciezka_danych: Path,
    *,
    twardo: bool = True,
) -> tuple[bool, str]:
    """
    Porownuje odcisk pliku danych z odciskiem zapisanym w konfiguracji.

    Przyjmuje:
        konfiguracja   - slownik z `wczytaj_konfiguracje`
        sciezka_danych - plik model.parquet
        twardo         - True podnosi wyjatek przy rozbieznosci, False zwraca ja
                         do decyzji wolajacego

    Zwraca:
        pare (czy_zgodny, odcisk_faktyczny).
    """
    oczekiwany = konfiguracja["dane"]["sha256"]
    faktyczny = policz_sha256(sciezka_danych)
    zgodny = faktyczny == oczekiwany
    if not zgodny and twardo:
        raise ValueError(
            f"{Path(sciezka_danych).name} ma odcisk {faktyczny[:12]}..., "
            f"a konfiguracja opisuje {oczekiwany[:12]}... - dane sie zmienily, "
            "wiec wyniki policzone na obu wersjach nie sa porownywalne"
        )
    return zgodny, faktyczny


def podzial_grupowy(
    grupy: Sequence[Any],
    n_podzialow: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Zwraca podzialy GroupKFold jako liste par tablic pozycyjnych.

    Zwracane tablice indeksuja pozycje wierszy, wiec w pandas siega sie po nie
    przez `.iloc`; `.loc` adresuje po etykiecie i dalby tu ciche bzdury.

    Przyjmuje:
        grupy       - wartosc grupujaca dla kazdego wiersza, zwykle player_id
        n_podzialow - liczba czesci

    Zwraca:
        liste par (indeksy uczace, indeksy walidacyjne).
    """
    grupy = np.asarray(grupy)
    unikalne = pd.unique(grupy)
    if len(unikalne) < n_podzialow:
        raise ValueError(
            f"{len(unikalne)} grup przy {n_podzialow} podzialach - za malo"
        )
    gkf = GroupKFold(n_splits=n_podzialow)
    # GroupKFold patrzy wylacznie na grupy, wiec X jest tu atrapa o wlasciwej
    # liczbie wierszy.
    atrapa = np.zeros((len(grupy), 1))
    return [(ucz, wal) for ucz, wal in gkf.split(atrapa, groups=grupy)]


def podsumowanie_foldow(
    grupy: Sequence[Any],
    n_podzialow: int = 5,
) -> pd.DataFrame:
    """
    Opisuje podzial GroupKFold: liczebnosci czesci i rozlacznosc grup.

    Przyjmuje:
        grupy       - wartosc grupujaca dla kazdego wiersza
        n_podzialow - liczba czesci

    Zwraca:
        ramke z wierszem na kazdy fold: liczba wierszy, liczba grup i liczba
        grup wspolnych z czescia uczaca (ma byc zerem).
    """
    grupy = np.asarray(grupy)
    wiersze = []
    for numer, (ucz, wal) in enumerate(podzial_grupowy(grupy, n_podzialow), start=1):
        wspolne = set(grupy[ucz]) & set(grupy[wal])
        wiersze.append(
            {
                "fold": numer,
                "n_ucz": len(ucz),
                "n_wal": len(wal),
                "grup_ucz": len(set(grupy[ucz])),
                "grup_wal": len(set(grupy[wal])),
                "grup_wspolnych": len(wspolne),
            }
        )
    return pd.DataFrame(wiersze)


def pary(ramka: pd.DataFrame, kolumny: Iterable[str]) -> set[tuple]:
    """
    Zwraca zbior krotek zbudowanych ze wskazanych kolumn.

    Krotka zamiast listy, bo tylko ona jest heszowalna, czyli moze trafic
    do zbioru i wziac udzial w przecieciu.

    Przyjmuje:
        ramka   - dowolna tabela
        kolumny - nazwy kolumn tworzacych klucz

    Zwraca:
        zbior krotek.
    """
    return set(map(tuple, ramka[list(kolumny)].to_numpy()))


def raport_przeciec(
    dane: pd.DataFrame,
    *,
    kolumna_podzialu: str = "podzial",
    klucz: Sequence[str] = ("player_id", "season"),
    grupa: str = "player_id",
) -> pd.DataFrame:
    """
    Zestawia przeciecia miedzy zbiorami: par klucza oraz samych grup.

    Przeciecie par musi byc puste. Przeciecie samych zawodnikow puste nie bedzie
    i nie ma byc - podzial jest czasowy, wiec ten sam zawodnik wraca w kolejnych
    sezonach. To swiadoma cecha podzialu, ktora raportujemy liczbowo (D-12).

    Przyjmuje:
        dane             - zbior modelowy z kolumna podzialu
        kolumna_podzialu - nazwa tej kolumny
        klucz            - kolumny tworzace klucz obserwacji
        grupa            - kolumna grupujaca, zwykle player_id

    Zwraca:
        ramke z wierszem na kazda pare zbiorow.
    """
    nazwy = sorted(dane[kolumna_podzialu].unique())
    klucze = {n: pary(dane[dane[kolumna_podzialu] == n], klucz) for n in nazwy}
    grupy = {n: set(dane.loc[dane[kolumna_podzialu] == n, grupa]) for n in nazwy}

    wiersze = []
    for i, a in enumerate(nazwy):
        for b in nazwy[i + 1 :]:
            wspolne_grupy = grupy[a] & grupy[b]
            wiersze.append(
                {
                    "zbior_a": a,
                    "zbior_b": b,
                    "wspolnych_par": len(klucze[a] & klucze[b]),
                    "wspolnych_grup": len(wspolne_grupy),
                    "udzial_grup_b": len(wspolne_grupy) / max(len(grupy[b]), 1),
                }
            )
    return pd.DataFrame(wiersze)
