"""Parametry przygotowania danych."""

from pathlib import Path


def znajdz_korzen_repo(start: Path) -> Path:
    """Idzie w gore katalogow, az znajdzie pyproject.toml."""
    for katalog in [start, *start.parents]:
        if (katalog / "pyproject.toml").exists():
            return katalog
    raise RuntimeError(f"nie znalazlem korzenia repozytorium, szukajac od {start}")


# Sciezki. data/raw tylko do odczytu.
KATALOG_REPO = znajdz_korzen_repo(Path(__file__).resolve())
SCIEZKA_DB = KATALOG_REPO / "data" / "raw" / "fbref" / "master.db"
SCIEZKA_TM = KATALOG_REPO / "data" / "raw" / "tm"
KATALOG_WYJSCIA = KATALOG_REPO / "data" / "processed"


# --- Zrodla FBref ---

# Bez tabeli Goalkeeper - bramkarzy pomijam (D-07).
TABELE_POLOWE = [
    "Summary",
    "Passing",
    "Possession",
    "Defensive_Actions",
    "Pass_Types",
    "Miscellaneous",
]

KLUCZ = ["match_id", "player_id"]
KLUCZ_SEZON = ["player_id", "season"]

# Attendance pomijam: 94% brakow w sezonie 2020-2021, bylaby znacznikiem pandemii.
KOLUMNY_Z_MATCH = ["match_id", "competition", "season", "date"]

# Kolumny identyczne z kolumnami w innych tabelach (sprawdzone na wszystkich wierszach).
KOLUMNY_ZREDUNDOWANE = {
    "Summary": [
        "assists",
        "blocks",
        "carries",
        "interceptions",
        "progressive_carries",
        "progressive_passes",
        "red_cards",
        "tackles",
        "yellow_cards",
        "touches",
        "passes_completed",
        "passes_attempted",
        "pass_completion_percentage",
        # w Summary te dwie sa zamienione miejscami, biore wersje z Possession
        "successful_dribbles",
        "dribbles_attempted",
        # Summary.xA to w rzeczywistosci xAG, takie samo jak Passing.xAG
        "xA",
    ],
    "Pass_Types": [
        "passes_completed",
        "passes_attempted",
    ],
    "Miscellaneous": [
        "crosses",
        "interceptions",
        "tackles_won",
    ],
}

# Dwie rozne metryki o tej samej nazwie: spalone z podan i spalone zawodnika.
ZMIANY_NAZW = {
    "Pass_Types": {"offsides": "offsides_z_podan"},
}


# --- Filtry zbioru ---

# Odrzucone: 2019-2020 (Ligue 1 279/380 meczow), 2024-2025 (25% brakow celu),
# 2025-2026 (sezon trwa). Patrz D-08.
SEZONY = [
    "2017-2018",
    "2018-2019",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
]

# Ponizej progu statystyki per 90 to szum (1 podanie w 10 minut = 9 na 90).
MIN_MINUT = 225


# --- Slowniki ---

MAPA_POZYCJI = {
    "CB": "DEF",
    "LB": "DEF",
    "RB": "DEF",
    "WB": "DEF",
    "DM": "MID",
    "CM": "MID",
    "LM": "MID",
    "RM": "MID",
    "AM": "MID",
    "LW": "FOR",
    "RW": "FOR",
    "FW": "FOR",
}

# Kraje CONMEBOL.
KODY_AMERYKA_PLD = frozenset(
    {
        "ARG",
        "BOL",
        "BRA",
        "CHI",
        "COL",
        "ECU",
        "PAR",
        "PER",
        "URU",
        "VEN",
    }
)

# Europa wedlug UEFA (stad m.in. ISR, KAZ, RUS). Kosowo ma w FBref kod KVX.
KODY_EUROPA = frozenset(
    {
        "ALB",
        "AND",
        "ARM",
        "AUT",
        "AZE",
        "BLR",
        "BEL",
        "BIH",
        "BUL",
        "CRO",
        "CYP",
        "CZE",
        "DEN",
        "ENG",
        "EST",
        "FRO",
        "FIN",
        "FRA",
        "GEO",
        "GER",
        "GIB",
        "GRE",
        "HUN",
        "ISL",
        "ISR",
        "ITA",
        "KAZ",
        "KVX",
        "LVA",
        "LIE",
        "LTU",
        "LUX",
        "MLT",
        "MDA",
        "MNE",
        "NED",
        "MKD",
        "NIR",
        "NOR",
        "POL",
        "POR",
        "IRL",
        "ROU",
        "RUS",
        "SMR",
        "SCO",
        "SRB",
        "SVK",
        "SVN",
        "ESP",
        "SWE",
        "SUI",
        "TUR",
        "UKR",
        "WAL",
    }
)


# --- Agregacja ---

# Wskaznik: (licznik, kolumny mianownika). Licze sume licznikow / sume mianownikow
# w sezonie, nie srednia z procentow z meczow.
# Jawna lista, bo tackled_perecentage ma w zrodle literowke.
WSKAZNIKI_PROCENTOWE = {
    "completion_percentage": ("total_completed", ["total_attempted"]),
    "short_completion_percentage": ("short_completed", ["short_attempted"]),
    "med_completion_percentage": ("med_completed", ["med_attempted"]),
    "long_completion_percentage": ("long_completed", ["long_attempted"]),
    "dribble_success_percentage": ("successful_dribbles", ["dribbles_attempted"]),
    "tackled_perecentage": ("tackled", ["dribbles_attempted"]),
    "successful_dribbler_tackle_percentage": (
        "dribblers_tackled",
        ["attempted_tackles_vs_dribblers"],
    ),
    "aerials_won_percentage": ("aerials_won", ["aerials_won", "aerials_lost"]),
}

# Kolumny, ktorych nie sumuje. Wszystko inne traktuje jako statystyke licznikowa.
KOLUMNY_NIELICZNIKOWE = [
    "match_id",
    "player_id",
    "season",
    "competition",
    "date",
    "home_away",
    "squad_number",
    "start",
    "nation",
    "position",
    "age",
    "name",
    "data_urodzenia",
    "wiek",
    "kod_kraju",
    "region",
    "minutes",
    "pozycja_mecz",
    # tworzone przy agregacji
    "minuty_sezon",
    "mecze",
    "mecze_od_poczatku",
    "liga",
    "pozycja",
    "zmienil_lige",
]


# --- Transfermarkt ---

# Wieksza roznica dat urodzenia FBref/TM oznacza innego zawodnika.
MAX_ROZNICA_DAT_DNI = 7

# Okno wyceny dla sezonu konczacego sie w roku t: 1.04-30.09, wycena najblizsza 30.06.
OKNO_WYCENY_OD = "-04-01"
OKNO_WYCENY_DO = "-09-30"
DZIEN_ODNIESIENIA = "-06-30"


# --- Podzial zbiorow (D-16) ---

SEZONY_TRENING = ["2017-2018", "2018-2019", "2020-2021", "2021-2022"]
SEZONY_KALIBRACJA = ["2022-2023"]
SEZONY_TEST = ["2023-2024"]

# Liga odlozona na pomiar przesuniecia dziedziny.
LIGA_ZBIOR_C = "Primeira_Liga"


# --- Indeks inflacji ---

PRZEDZIALY_WIEKU = [0, 20, 22, 24, 26, 28, 30, 32, 99]
MIN_OBSERWACJI_KOMORKA = 20
