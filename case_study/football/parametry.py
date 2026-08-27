"""
Parametry przygotowania danych.

"""

from pathlib import Path


def znajdz_korzen_repo(start: Path) -> Path:
    """
    Idzie w gore katalogow, az znajdzie pyproject.toml.

    """
    for katalog in [start, *start.parents]:
        if (katalog / "pyproject.toml").exists():
            return katalog
    raise RuntimeError(f"nie znalazlem korzenia repozytorium, szukajac od {start}")


# Sciezki. data/raw jest tylko do odczytu - nic tam nie zapisujemy.
KATALOG_REPO = znajdz_korzen_repo(Path(__file__).resolve())
SCIEZKA_DB = KATALOG_REPO / "data" / "raw" / "fbref" / "master.db"
SCIEZKA_TM = KATALOG_REPO / "data" / "raw" / "tm"
KATALOG_WYJSCIA = KATALOG_REPO / "data" / "processed"


# --- Zrodla FBref ---

# Tabele statystyk polowych. Goalkeeper pomijamy, bo odrzucamy bramkarzy.
TABELE_POLOWE = [
    "Summary",
    "Passing",
    "Possession",
    "Defensive_Actions",
    "Pass_Types",
    "Miscellaneous",
]

# Klucze: wystep w meczu i para zawodnik-sezon.
KLUCZ = ["match_id", "player_id"]
KLUCZ_SEZON = ["player_id", "season"]

# Z tabeli Match bierzemy tylko kontekst zawodnika. Reszta opisuje mecz.
# Attendance swiadomie pomijamy: 94% brakow w sezonie 2020-2021, braki
# skorelowane z sezonem, wiec cecha bylaby znacznikiem pandemii.
KOLUMNY_Z_MATCH = ["match_id", "competition", "season", "date"]

# FBref powiela te same metryki w kilku tabelach. Sprawdzono wartosc po wartosci
# na wszystkich wierszach - ponizsze sa identyczne z odpowiednikiem w tabeli
# szczegolowej i zostaja usuniete. Zostawienie ich dawaloby pary cech
# skorelowanych w 100%, co psuje ranking waznosci cech (W4).
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
        # Uwaga: te dwie sa w Summary zamienione miejscami (udane > probowane
        # w 173 tys. wierszy). Bierzemy wersje z Possession.
        "successful_dribbles",
        "dribbles_attempted",
        # Summary.xA to w rzeczywistosci xAG - zgadza sie z Passing.xAG w 100%.
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

# Kolizja nazw, ktora nie jest redundancja - dwie rozne metryki:
#   Pass_Types.offsides    - podania zagrane na spalonego (wina podajacego)
#   Miscellaneous.offsides - spalone zlapane przez zawodnika (wina przyjmujacego)
ZMIANY_NAZW = {
    "Pass_Types": {"offsides": "offsides_z_podan"},
}


# --- Filtry zbioru ---

# Z dziewieciu dostepnych sezonow zostaje szesc:
#   2019-2020 - Ligue 1 rozegrala 279 z 380 meczow (pandemia)
#   2024-2025 - 25% brakow zmiennej celu, skorelowanych z liga
#   2025-2026 - sezon w trakcie
SEZONY = [
    "2017-2018",
    "2018-2019",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
]

# Prog minut w sezonie. Ponizej statystyki per 90 sa szumem: jedno podanie
# w 10 minut daje 9 podan na 90. Parametr do analizy wrazliwosci - przy 225
# zostaje 81% par, przy 450 okolo 74%, przy 900 okolo 60%.
MIN_MINUT = 225


# --- Slowniki ---

# Pozycje FBref sprowadzone do trzech klas. Bramkarzy (GK) odrzucamy - ocenia
# sie ich innymi metrykami, ktorych zawodnicy z pola nie maja.
MAPA_POZYCJI = {
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "WB": "DEF",
    "DM": "MID", "CM": "MID", "LM": "MID", "RM": "MID", "AM": "MID",
    "LW": "FOR", "RW": "FOR", "FW": "FOR",
}

# Ameryka Poludniowa jako osobny region - rynek wycenia tych zawodnikow inaczej
# (premia za potencjal, koszt adaptacji). Lista CONMEBOL.
KODY_AMERYKA_PLD = frozenset({
    "ARG", "BOL", "BRA", "CHI", "COL", "ECU", "PAR", "PER", "URU", "VEN",
})

# Europa wedlug UEFA, nie geografii - to podzial rynku pilkarskiego. Stad ISR,
# ARM, AZE, GEO, KAZ i RUS. Wielka Brytania to cztery federacje, a Kosowo ma
# w FBref kod KVX (nie KOS).
KODY_EUROPA = frozenset({
    "ALB", "AND", "ARM", "AUT", "AZE", "BLR", "BEL", "BIH", "BUL", "CRO",
    "CYP", "CZE", "DEN", "ENG", "EST", "FRO", "FIN", "FRA", "GEO", "GER",
    "GIB", "GRE", "HUN", "ISL", "ISR", "ITA", "KAZ", "KVX", "LVA", "LIE",
    "LTU", "LUX", "MLT", "MDA", "MNE", "NED", "MKD", "NIR", "NOR", "POL",
    "POR", "IRL", "ROU", "RUS", "SMR", "SCO", "SRB", "SVK", "SVN", "ESP",
    "SWE", "SUI", "TUR", "UKR", "WAL",
})


# --- Agregacja ---

# Wskaznik procentowy: nazwa -> (licznik, lista kolumn mianownika).
#
# Procentow nie usredniamy z poziomu meczu. Sumujemy liczniki i mianowniki,
# wskaznik liczymy dopiero na poziomie sezonu. Inaczej mecz z dwiema probami
# wazylby tyle samo co mecz z osmioma: 1/2 i 0/8 daje srednio 25%, a poprawnie
# 1/10 = 10%. Zmierzony blad naiwnej sredniej siega 33 p.p.
#
# Kazda para zweryfikowana na danych - odtwarza procent FBref w 100% wierszy.
# Uwaga: tackled_perecentage ma literowke w schemacie zrodlowym, wiec nie
# znajdzie jej zadne wyszukiwanie po ciagu "percent". Stad jawna lista.
WSKAZNIKI_PROCENTOWE = {
    "completion_percentage":       ("total_completed", ["total_attempted"]),
    "short_completion_percentage": ("short_completed", ["short_attempted"]),
    "med_completion_percentage":   ("med_completed",   ["med_attempted"]),
    "long_completion_percentage":  ("long_completed",  ["long_attempted"]),
    "dribble_success_percentage":  ("successful_dribbles", ["dribbles_attempted"]),
    "tackled_perecentage":         ("tackled",             ["dribbles_attempted"]),
    "successful_dribbler_tackle_percentage": (
        "dribblers_tackled", ["attempted_tackles_vs_dribblers"]
    ),
    # Mianownik z dwoch kolumn - FBref nie podaje "pojedynkow rozegranych".
    "aerials_won_percentage":      ("aerials_won", ["aerials_won", "aerials_lost"]),
}

# Kolumny, ktore nie sa statystyka licznikowa. Statystyki do zsumowania
# wyznaczamy jako "cala reszta" - gdyby FBref dolozyl nowa metryke, zostanie
# zsumowana automatycznie zamiast po cichu wypasc.
KOLUMNY_NIELICZNIKOWE = [
    "match_id", "player_id", "season", "competition", "date",
    "home_away", "squad_number", "start",
    "nation", "position", "age",
    "name", "data_urodzenia", "wiek", "kod_kraju", "region",
    "minutes", "pozycja_mecz",
    # utworzone przy agregacji - podsumowania sezonu, nie statystyki gry
    "minuty_sezon", "mecze", "mecze_od_poczatku", "liga", "pozycja",
    "zmienil_lige",
]


# --- Transfermarkt ---

# Prog zgodnosci dat urodzenia przy weryfikacji crosswalku. Rozbieznosc powyzej
# tygodnia oznacza innego zawodnika, czyli cudza wycene w wierszu. Ponizej -
# drobna niescislosc zrodla. Test uzywa tylko dat i nie dotyka celu, wiec
# odrzucenie na jego podstawie nie jest wyciekiem.
MAX_ROZNICA_DAT_DNI = 7

# Okno wyceny dla sezonu konczacego sie w roku t: bierzemy obserwacje z okresu
# 1.04 - 30.09 i wybieramy te najblizsza 30 czerwca. Nie maksimum - maksimum
# z kilku obserwacji jest estymatorem obciazonym w gore i premiowaloby
# zawodnikow czesciej wycenianych. Zapisane jako koncowki dat doklejane do roku.
OKNO_WYCENY_OD = "-04-01"
OKNO_WYCENY_DO = "-09-30"
DZIEN_ODNIESIENIA = "-06-30"


# --- Podzial zbiorow ---

# Podzial czasowy, nie losowy: uczymy sie na przeszlosci, oceniamy na nastepnym
# sezonie. Kalibracja jest osobna, bo predykcja konformalna kalibrowana na
# treningu daje zawyzone pokrycie (W3).
SEZONY_TRENING = ["2017-2018", "2018-2019", "2020-2021", "2021-2022"]
SEZONY_KALIBRACJA = ["2022-2023"]
SEZONY_TEST = ["2023-2024"]

# Liga odlozona w calosci na pomiar przesuniecia dziedziny (W2). Nie moze
# wystapic w treningu ani kalibracji.
LIGA_ZBIOR_C = "Primeira_Liga"


# --- Indeks inflacji ---

# Przedzialy wieku do porownan o stalym skladzie - zeby efekt starzenia sie
# nie mieszal z ruchem cen.
PRZEDZIALY_WIEKU = [0, 20, 22, 24, 26, 28, 30, 32, 99]

# Komorki mniejsze niz tyle obserwacji pomijamy.
MIN_OBSERWACJI_KOMORKA = 20
