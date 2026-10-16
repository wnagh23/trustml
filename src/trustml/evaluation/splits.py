"""
Protokol podzialu: odczyt configs/split.yaml, kontrola odcisku danych
i GroupKFold po zawodniku do strojenia na treningu.
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
    """Liczy SHA-256 pliku, czytajac go porcjami."""
    skrot = hashlib.sha256()
    with Path(sciezka).open("rb") as plik:
        while porcja := plik.read(kawalek):
            skrot.update(porcja)
    return skrot.hexdigest()


def wczytaj_konfiguracje(sciezka: Path) -> dict[str, Any]:
    """Wczytuje split.yaml do slownika."""
    sciezka = Path(sciezka)
    if not sciezka.exists():
        raise FileNotFoundError(f"brak konfiguracji podzialu: {sciezka}")
    return yaml.safe_load(sciezka.read_text(encoding="utf-8"))


def sprawdz_odcisk_danych(
    konfiguracja: dict[str, Any],
    sciezka_danych: Path,
    *,
    twardo: bool = True,
) -> tuple[bool, str]:
    """
    Porownuje SHA-256 pliku danych z wpisem w konfiguracji.

    Przy `twardo=True` rozbieznosc rzuca wyjatek. Zwraca (czy_zgodny, odcisk).
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
    Zwraca foldy GroupKFold jako liste par (indeksy uczace, walidacyjne).

    Indeksy sa pozycyjne, wiec w pandas trzeba uzyc .iloc, nie .loc.
    """
    grupy = np.asarray(grupy)
    unikalne = pd.unique(grupy)
    if len(unikalne) < n_podzialow:
        raise ValueError(
            f"{len(unikalne)} grup przy {n_podzialow} podzialach - za malo"
        )
    gkf = GroupKFold(n_splits=n_podzialow)
    # GroupKFold patrzy tylko na grupy, X jest atrapa o dobrej liczbie wierszy
    atrapa = np.zeros((len(grupy), 1))
    return [(ucz, wal) for ucz, wal in gkf.split(atrapa, groups=grupy)]


def podsumowanie_foldow(
    grupy: Sequence[Any],
    n_podzialow: int = 5,
) -> pd.DataFrame:
    """Tabela z liczebnosciami foldow i liczba grup wspolnych (ma byc 0)."""
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
    """Zbior krotek z wartosci wskazanych kolumn."""
    return set(map(tuple, ramka[list(kolumny)].to_numpy()))


def raport_przeciec(
    dane: pd.DataFrame,
    *,
    kolumna_podzialu: str = "podzial",
    klucz: Sequence[str] = ("player_id", "season"),
    grupa: str = "player_id",
) -> pd.DataFrame:
    """
    Przeciecia miedzy kazda para zbiorow: wspolne pary klucza i wspolni zawodnicy.

    Wspolnych par ma byc 0. Wspolni zawodnicy sa normalni przy podziale czasowym (D-12).
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
