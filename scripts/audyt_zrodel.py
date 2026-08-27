"""
Audyt archiwow ZIP z surowymi stronami FBref.

master.db przychodzi bez metadanych: bez wersji, daty scrape'u i adresu
zrodlowego. Archiwa ZIP obok bazy trzymaja surowe strony, z ktorych ja zbudowano,
a te maja i adres kanoniczny, i date pobrania w naglowku pliku.

Skrypt czyta wylacznie metadane archiwow i probke stron, zostawiajac data/raw
w stanie nietknietym.

    python scripts/audyt_zrodel.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import zipfile
from datetime import UTC, datetime
from pathlib import Path

KATALOG_REPO = Path(__file__).resolve().parents[1]
KATALOG_ZIP = KATALOG_REPO / "data" / "raw" / "fbref"
SCIEZKA_DB = KATALOG_ZIP / "master.db"
WYJSCIE = KATALOG_REPO / "docs" / "zrodla_fbref.json"

# Adres kanoniczny w naglowku strony - jedyne miejsce, gdzie FBref zapisuje,
# skad pochodzi dokument.
WZOR_URL = re.compile(r"<link[^>]+rel=[\"']canonical[\"'][^>]+href=[\"']([^\"']+)")


def zbadaj_archiwum(sciezka: Path) -> dict:
    """
    Zbiera metadane jednego archiwum: liczbe stron, sezony i daty pobrania.

    Przyjmuje:
        sciezka - plik .zip

    Zwraca:
        slownik z podsumowaniem archiwum.
    """
    with zipfile.ZipFile(sciezka) as archiwum:
        wpisy = [w for w in archiwum.infolist() if not w.is_dir()]

        # Sciezka w archiwum wyglada jak "Bundesliga/2017-2018/01024937".
        sezony: dict[str, int] = {}
        for wpis in wpisy:
            czesci = Path(wpis.filename).parts
            if len(czesci) >= 2:
                sezony[czesci[1]] = sezony.get(czesci[1], 0) + 1

        # Format ZIP zapisuje czas lokalny bez strefy - doklejenie tu UTC byloby
        # dopisaniem informacji spoza pliku.
        daty = [datetime(*w.date_time) for w in wpisy]  # noqa: DTZ001

        # Adres bierzemy z pierwszej strony - wzorzec jest ten sam dla wszystkich.
        with archiwum.open(wpisy[0].filename) as plik:
            poczatek = plik.read(200_000).decode("utf-8", errors="ignore")
        trafienie = WZOR_URL.search(poczatek)
        przyklad_url = trafienie.group(1) if trafienie else None

        identyfikatory = {Path(w.filename).name for w in wpisy}

    return {
        "plik": sciezka.name,
        "rozmiar_mb": round(sciezka.stat().st_size / 1024 / 1024, 1),
        "stron": len(wpisy),
        "sezony": dict(sorted(sezony.items())),
        "scrape_od": min(daty).isoformat(timespec="seconds"),
        "scrape_do": max(daty).isoformat(timespec="seconds"),
        "przyklad_url": przyklad_url,
        "identyfikatory": identyfikatory,
    }


def main() -> None:
    archiwa = sorted(KATALOG_ZIP.glob("*.zip"))
    assert archiwa, f"nie znalazlem archiwow w {KATALOG_ZIP}"

    print(f"audyt {len(archiwa)} archiwow z {KATALOG_ZIP}\n")

    wyniki = []
    wszystkie_id: set[str] = set()
    for sciezka in archiwa:
        wynik = zbadaj_archiwum(sciezka)
        wszystkie_id |= wynik.pop("identyfikatory")
        wyniki.append(wynik)
        print(
            f"  {wynik['plik']:22s} {wynik['stron']:5d} stron  "
            f"{wynik['rozmiar_mb']:6.1f} MB  "
            f"scrape {wynik['scrape_od'][:10]} - {wynik['scrape_do'][:10]}"
        )

    print(f"\n  lacznie stron: {len(wszystkie_id)}")

    # Kontrola spojnosci: czy identyfikatory ze scrape'ow pokrywaja mecze w bazie.
    with sqlite3.connect(
        f"file:{SCIEZKA_DB.as_posix()}?mode=ro", uri=True
    ) as polaczenie:
        w_bazie = {r[0] for r in polaczenie.execute("SELECT match_id FROM Match")}

    print("\n  zgodnosc scrape'ow z master.db:")
    print(f"    meczow w bazie:            {len(w_bazie)}")
    print(f"    stron w archiwach:         {len(wszystkie_id)}")
    print(f"    w bazie, brak strony:      {len(w_bazie - wszystkie_id)}")
    print(f"    strona jest, brak w bazie: {len(wszystkie_id - w_bazie)}")

    podsumowanie = {
        "opis": (
            "Prowieniencja master.db odtworzona z archiwow ZIP. Kazda strona to "
            "surowy dokument FBref o adresie https://fbref.com/en/matches/<match_id>/..., "
            "a data modyfikacji pliku w archiwum to data pobrania."
        ),
        "zbadano": datetime.now(UTC).isoformat(timespec="seconds"),
        "wzor_adresu": "https://fbref.com/en/matches/{match_id}/",
        "scrape_od": min(w["scrape_od"] for w in wyniki),
        "scrape_do": max(w["scrape_do"] for w in wyniki),
        "stron_lacznie": len(wszystkie_id),
        "meczow_w_master_db": len(w_bazie),
        "stron_bez_meczu_w_bazie": len(wszystkie_id - w_bazie),
        "meczow_bez_strony": len(w_bazie - wszystkie_id),
        "archiwa": wyniki,
    }

    WYJSCIE.parent.mkdir(parents=True, exist_ok=True)
    WYJSCIE.write_text(
        json.dumps(podsumowanie, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  zapisano: {WYJSCIE.relative_to(KATALOG_REPO)}")


if __name__ == "__main__":
    main()
