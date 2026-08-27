"""
Przygotowanie zbioru do modelowania predykcyjnego:
    1. Dane FBref (statystyki zawodników mecz po meczu)
    2. Transfermarkt (wycena zawodnika na dzień 30. czerwca, dzień zakończenia danego sezonu)

Wyjściem skryptu powinien być jeden wiersz na parę (zawodnik, sezon) z kolumnami dotyczącymi
atrybutów piłkarza w danym sezonie oraz jego wartości rynkowej

Działanie skryptu podzielono na 10 etapow:

ETAP 1  wczytanie i deduplikacja tabel z master.db
ETAP 2  polaczenie osmiu tabel po (match_id, player_id)
ETAP 3  parsowanie: wiek, narodowosc, pozycja
ETAP 4  agregacja mecz do zawodnik x sezon
ETAP 5  wskazniki procentowe i normalizacja per 90 minut
ETAP 6  crosswalk do Transfermarktu i cechy statyczne
ETAP 7  dolaczenie wyceny (okno IV-IX, najblizsza 30.06)
ETAP 8  podzial na trening / kalibracja / test / zbior_C
ETAP 9  deflacja (indeks liczony tylko na treningu)
ETAP 10 zapis plikow i manifestu     

"""

from __future__ import annotations

import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# PARAMETRY

# Sciezki

KATALOG_REPO = Path(__file__).resolve().parents[2]
SCIEZKA_DB = KATALOG_REPO / "data" / "raw" / "fbref" / "master.db"
SCIEZKA_TM = KATALOG_REPO / "data" / "raw" / "tm"
KATALOG_WYJSCIA = KATALOG_REPO / "data" / "processed"

# Tabele statystyk polowych

TABELE_POLOWE = [
    "Summary",
    "Passing",
    "Possession",
    "Defensive_Actions",
    "Pass_Types",
    "Miscellaneous",
]

# Klucz zlozony

KLUCZ = ["match_id", "player_id"]


# Wybieramy tylko te kolumny, ktore opisuja zawodnika - tabela match ma tez info dot. meczu,
# ale tego nie potrzebujemy.

KOLUMNY_Z_MATCH = ["match_id", "competition", "season", "date"]



# Usuwamy kolumny redundantne - FBRef powiela te same metryki w kilku tabelach.
# Sprawdzono powtarzalnosc danych w wszystkich wierszach. 

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
        "successful_dribbles",
        "dribbles_attempted",
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


# Jedna kolizja nazw niebędąca redundancja:

#   Pass_Types.offsides    - podania zagrane na spalonego (wina podajacego)
#   Miscellaneous.offsides - spalone zlapane przez zawodnika (wina przyjmujacego)

ZMIANY_NAZW = {
    "Pass_Types": {"offsides": "offsides_z_podan"},
}

# Sezony wchodzace do analizy. Z dziewieciu dostepnych zostaje szesc:
#   2019-2020 - Ligue 1 rozegrala 279 z 380 meczow
#   2024-2025 - 25% brakow zmiennej celu, skorelowanych z liga
#   2025-2026 - sezon w trakcie, dane niekompletne z definicji

SEZONY = [
    "2017-2018",
    "2018-2019",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
]

# Mapowanie pozycji FBref na trzy klasy. Sprowadzamy pozycje do klas:
#  DEF - obroncy (CB, LB, RB, WB)
#  MID - pomocnicy (DM, CM, LM, RM, AM)
#  FOR - napastnicy (LW, RW, FW)
#  Bramkarze są odrzucani

MAPA_POZYCJI = {
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "WB": "DEF",
    "DM": "MID", "CM": "MID", "LM": "MID", "RM": "MID", "AM": "MID",
    "LW": "FOR", "RW": "FOR", "FW": "FOR",
}

# Kraje CONMEBOL - Ameryka Poludniowa jako osobny region. Zakładamy że 
# rynek transferowy wycenia zawodnikow z Ameryki Poludniowej inaczej

KODY_AMERYKA_PLD = frozenset({
    "ARG", "BOL", "BRA", "CHI", "COL", "ECU", "PAR", "PER", "URU", "VEN",
})

# Kraje UEFA - pelna lista 55 federacji, zakladamy że rynek wycenia europejczykow
# inaczej

KODY_EUROPA = frozenset({
    "ALB", "AND", "ARM", "AUT", "AZE", "BLR", "BEL", "BIH", "BUL", "CRO",
    "CYP", "CZE", "DEN", "ENG", "EST", "FRO", "FIN", "FRA", "GEO", "GER",
    "GIB", "GRE", "HUN", "ISL", "ISR", "ITA", "KAZ", "KVX", "LVA", "LIE",
    "LTU", "LUX", "MLT", "MDA", "MNE", "NED", "MKD", "NIR", "NOR", "POL",
    "POR", "IRL", "ROU", "RUS", "SMR", "SCO", "SRB", "SVK", "SVN", "ESP",
    "SWE", "SUI", "TUR", "UKR", "WAL",
})


# ETAP 4-5: agregacja do sezonu, normalizacja predyktorow per 90 minut i wskazniki procentowe

# Klucz zbioru docelowego

KLUCZ_SEZON = ["player_id", "season"]

# Minimalny prog minut w sezonie

MIN_MINUT = 225


# Prog zgodnosci dat urodzenia przy weryfikacji crosswalku (w dniach).

MAX_ROZNICA_DAT_DNI = 7


# ETAP 7: moment pomiaru zmiennej celu

# Dla sezonu konczacego sie w roku t bierzemy wycene z okna 1.04 - 30.09 tego
# roku i wybieramy te o dacie najblizszej 30 czerwca. 

# Zapisane jako koncowki dat, doklejane do roku ("2024" + "-06-30").
OKNO_WYCENY_OD = "-04-01"
OKNO_WYCENY_DO = "-09-30"
DZIEN_ODNIESIENIA = "-06-30"


# ETAP 8: podzial zbiorow

# Podzial jest czasowy - uczymy sie na przeszlosci, oceniamy na nastepnym sezonie

SEZONY_TRENING = ["2017-2018", "2018-2019", "2020-2021", "2021-2022"]
SEZONY_KALIBRACJA = ["2022-2023"]
SEZONY_TEST = ["2023-2024"]

# Liga odlozona w calosci na pomiar przesuniecia dziedziny

LIGA_ZBIOR_C = "Primeira_Liga"


# ETAP 9: indeks inflacji rynku

# Granice przedzialow wiekowych. Porownujemy zawodnikow w tym samym wieku,
# zeby efekt starzenia sie nie mieszal z ruchem cen.
PRZEDZIALY_WIEKU = [0, 20, 22, 24, 26, 28, 30, 32, 99]

# Komorki mniejsze niz tyle obserwacji pomijamy - mediana z kilku wartosci
# jest zbyt niestabilna.
MIN_OBSERWACJI_KOMORKA = 20


# Wskazniki procentowe

# procentow nie usredniamy z poziomu
# meczu. Agregujemy liczniki i mianowniki osobno, a wskaznik liczymy dopiero
# na poziomie sezonu jako suma_licznikow / suma_mianownikow.

WSKAZNIKI_PROCENTOWE = {
    # celnosc podan - ogolem i w podziale na dystans
    "completion_percentage":       ("total_completed", ["total_attempted"]),
    "short_completion_percentage": ("short_completed", ["short_attempted"]),
    "med_completion_percentage":   ("med_completed",   ["med_attempted"]),
    "long_completion_percentage":  ("long_completed",  ["long_attempted"]),
    # dryblingi zawodnika: udane oraz zatrzymane przez rywala
    "dribble_success_percentage":  ("successful_dribbles", ["dribbles_attempted"]),
    "tackled_perecentage":         ("tackled",             ["dribbles_attempted"]),
    # skutecznosc w odbieraniu pilki dryblujacemu rywalowi
    "successful_dribbler_tackle_percentage": (
        "dribblers_tackled", ["attempted_tackles_vs_dribblers"]
    ),
    # pojedynki powietrzne
    "aerials_won_percentage":      ("aerials_won", ["aerials_won", "aerials_lost"]),
}


KOLUMNY_NIELICZNIKOWE = [
    # klucze i identyfikatory
    "match_id", "player_id", "season", "competition", "date",
    # pola opisujace pojedynczy mecz
    "home_away", "squad_number", "start",
    # pola tekstowe przetworzone
    "nation", "position", "age",
    # cechy stale zawodnika
    "name", "data_urodzenia", "wiek", "kod_kraju", "region",
    # obslugiwane osobno
    "minutes", "pozycja_mecz",
    # kolumny utworzone w ETAPIE 4 - podsumowania sezonu, nie statystyki gry.
    # Bez nich na liscie ETAP 5 probowalby liczyc "mecze na 90 minut".
    "minuty_sezon", "mecze", "mecze_od_poczatku", "liga", "pozycja",
    "zmienil_lige",
]



# ETAP 1  wczytanie i deduplikacja tabel z master.db


def wczytaj_tabele(polaczenie: sqlite3.Connection, nazwa: str) -> pd.DataFrame:
    """
    Wczytuje jedna tabele z master.db i usuwa z niej zdublowane wiersze.

    Przyjmuje:
        polaczenie - otwarte polaczenie sqlite3 do master.db
        nazwa      - nazwa tabeli, np. "Summary"

    Zwraca:
        DataFrame o unikalnym kluczu (match_id, player_id).

    """
    
    tabela = pd.read_sql(f'SELECT * FROM "{nazwa}"', polaczenie)
    przed = len(tabela)


    tabela = tabela.drop_duplicates()
    usuniete = przed - len(tabela)

    if usuniete:
        print(f"  {nazwa:20s} usunieto {usuniete} zdublowanych wierszy")

    
    klucz = [k for k in KLUCZ if k in tabela.columns]

    
    nadmiarowe = tabela.duplicated(subset=klucz).sum()
    assert nadmiarowe == 0, (
        f"{nazwa}: po deduplikacji nadal {nadmiarowe} zdublowanych kluczy "
        f"{klucz} -duplikaty nie byly identycznymi kopiami "
    )
    return tabela


def usun_redundancje(tabele: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Usuwa kolumny powielone miedzy tabelami i ujednoznacznia kolidujace nazwy.

    Przyjmuje:
        tabele - slownik {nazwa_tabeli: DataFrame} ze zdeduplikowanymi wierszami

    Zwraca:
        ten sam slownik, ale z wezszymi tabelami. Po tej operacji zadna nazwa
        kolumny (poza kluczem) nie powtarza sie miedzy tabelami polowymi.
    """
    for nazwa, do_usuniecia in KOLUMNY_ZREDUNDOWANE.items():
        tabele[nazwa] = tabele[nazwa].drop(columns=do_usuniecia)
        print(f"  {nazwa:20s} usunieto {len(do_usuniecia)} zredundowanych kolumn")

    for nazwa, mapowanie in ZMIANY_NAZW.items():
        tabele[nazwa] = tabele[nazwa].rename(columns=mapowanie)
        for stara, nowa in mapowanie.items():
            print(f"  {nazwa:20s} zmiana nazwy: {stara} -> {nowa}")

    # Sprawdzenie czy na pewno nie zostala zadna kolizja nazw
    # budujemy slownik {nazwa_kolumny: [tabele, w ktorych wystepuje]} wymuszamy,
    # zeby kazda lista miala dlugosc 1.

    gdzie_wystepuje: dict[str, list[str]] = {}
    for nazwa in TABELE_POLOWE:
        for kolumna in tabele[nazwa].columns:
            if kolumna not in KLUCZ:
                gdzie_wystepuje.setdefault(kolumna, []).append(nazwa)
    kolizje = {k: v for k, v in gdzie_wystepuje.items() if len(v) > 1}
    assert not kolizje, f"pozostaly kolizje nazw kolumn miedzy tabelami: {kolizje}"
    print(f"  OK: zadna z {len(gdzie_wystepuje)} nazw kolumn nie powtarza sie miedzy tabelami")

    return tabele


def etap_1_wczytaj_i_zdeduplikuj(sciezka_db: Path) -> dict[str, pd.DataFrame]:
    """
    Przyjmuje:
        sciezka_db - sciezka do pliku master.db (otwierany tylko do odczytu)

    Zwraca:
        slownik {nazwa_tabeli: DataFrame}
    """
    print("=" * 70)
    print("ETAP 1  wczytanie i deduplikacja tabel z master.db")
    print("=" * 70)

    assert sciezka_db.exists(), f"nie znaleziono bazy: {sciezka_db}"

    
    uri = f"file:{sciezka_db.as_posix()}?mode=ro"
    tabele: dict[str, pd.DataFrame] = {}

    
    from contextlib import closing

    with closing(sqlite3.connect(uri, uri=True)) as polaczenie:
        print("\n -- wczytywanie i deduplikacja wierszy --")
        for nazwa in [*TABELE_POLOWE, "Player_Info", "Match"]:
            tabele[nazwa] = wczytaj_tabele(polaczenie, nazwa)

        print("\n-- usuwanie kolumn powielonych miedzy tabelami --")
        tabele = usun_redundancje(tabele)

    
    print("\n-- stan po ETAPIE 1 --")
    print(f"  {'tabela':22s} {'wierszy':>10s} {'kolumn':>8s}")
    for nazwa, tabela in tabele.items():
        print(f"  {nazwa:22s} {len(tabela):10d} {tabela.shape[1]:8d}")

    assert tabele["Match"]["match_id"].is_unique, "match_id nie jest unikalny w Match"

    
    liczby = {n: len(tabele[n]) for n in [*TABELE_POLOWE, "Player_Info"]}
    assert len(set(liczby.values())) == 1, (
        f"tabele maja rozne liczby wierszy, wiec nie opisuja tego samego zbioru "
        f"wystepow: {liczby}"
    )
    print(f"\n  wszystkie tabele wystepow maja po {len(tabele['Summary'])} wierszy")

    return tabele



# ETAP 2  polaczenie tabel po (match_id, player_id)


def etap_2_polacz_tabele(tabele: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """


    Przyjmuje:
        tabele - slownik z ETAPU 1 (zdeduplikowany, bez kolizji nazw kolumn)

    Zwraca:
        DataFrame o unikalnym kluczu (match_id, player_id).

    """
    print("\n" + "=" * 70)
    print("ETAP 2  polaczenie tabel po (match_id, player_id)")
    print("=" * 70)

    # Zaczynamy od Player_Info, bo to ona mowi, kto zagral - reszta tabel dodaje
    # tylko, jak zagral.

    polaczone = tabele["Player_Info"]
    oczekiwane = len(polaczone)
    print(f"\n  start: Player_Info, {oczekiwane} wystepow, {polaczone.shape[1]} kolumn")

    for nazwa in TABELE_POLOWE:
        
        przed = polaczone.shape[1]
        polaczone = polaczone.merge(
            tabele[nazwa], on=KLUCZ, how="inner", validate="one_to_one"
        )

        assert len(polaczone) == oczekiwane, (
            f"po dolaczeniu {nazwa} jest {len(polaczone)} wierszy zamiast "
            f"{oczekiwane} - zbiory kluczy przestaly byc identyczne"
        )
        print(
            f"  + {nazwa:20s} {polaczone.shape[1] - przed:3d} nowych kolumn "
            f"-> {polaczone.shape[1]:3d} kolumn, {len(polaczone)} wierszy"
        )


    przed = polaczone.shape[1]
    polaczone = polaczone.merge(
        tabele["Match"][KOLUMNY_Z_MATCH],
        on="match_id",
        how="inner",
        validate="many_to_one",
    )
    assert len(polaczone) == oczekiwane, (
        f"po dolaczeniu Match jest {len(polaczone)} wierszy zamiast {oczekiwane} "
    )
    print(
        f"  + {'Match':20s} {polaczone.shape[1] - przed:3d} nowych kolumn "
        f"-> {polaczone.shape[1]:3d} kolumn, {len(polaczone)} wierszy"
    )

   
    sufiksy = [c for c in polaczone.columns if c.endswith(("_x", "_y"))]
    assert not sufiksy, f"laczenie wyprodukowalo zdublowane kolumny: {sufiksy}"

    nadmiarowe = polaczone.duplicated(subset=KLUCZ).sum()
    assert nadmiarowe == 0, f"{nadmiarowe} zdublowanych par (match_id, player_id)"

    print("\n-- stan po ETAPIE 2 --")
    print(f"  wystepow (wiersz = zawodnik w meczu): {len(polaczone)}")
    print(f"  kolumn:                               {polaczone.shape[1]}")
    print(f"  unikalnych zawodnikow:                {polaczone['player_id'].nunique()}")
    print(f"  unikalnych meczow:                    {polaczone['match_id'].nunique()}")
    print(f"  sezonow:                              {polaczone['season'].nunique()}")
    print(f"  rozgrywek:                            {polaczone['competition'].nunique()}")
    print("  OK: zero sufiksow _x/_y, klucz nadal unikalny")

    return polaczone


# ETAP 3  parsowanie: wiek, narodowosc, pozycja


def wybierz_najczestsza(
    ramka: pd.DataFrame, kolumna_klucza: str, kolumna_wartosci: str
) -> pd.Series:
    """
    Dla kazdego klucza wybiera najczesciej wystepujaca wartosc (dominanta).

    Przyjmuje:
        ramka            - dane z kolumna klucza i kolumna wartosci
        kolumna_klucza   - po czym grupujemy, np. "player_id"
        kolumna_wartosci - z czego wybieramy dominante, np. "data_urodzenia"

    Zwraca:
        Series indeksowana kluczem, jedna wartosc na klucz.
    """
    
    licznik = (
        ramka.groupby([kolumna_klucza, kolumna_wartosci]).size().reset_index(name="ile")
    )

    licznik = licznik.sort_values(
        [kolumna_klucza, "ile", kolumna_wartosci], ascending=[True, False, True]
    )

    najczestsze = licznik.drop_duplicates(subset=kolumna_klucza, keep="first")

    return najczestsze.set_index(kolumna_klucza)[kolumna_wartosci]


def wylicz_daty_urodzenia(wystepy: pd.DataFrame) -> pd.Series:
    """
    Odtwarza date urodzenia kazdego zawodnika z wieku podanego na dzien meczu.

    

    Przyjmuje:
        wystepy - tabela z kolumnami player_id, age, date

    Zwraca:
        Series: player_id -> data urodzenia (typ datetime).
        Zawodnicy bez ani jednego meczu z poprawnym wiekiem sie w niej nie znajda.
    """
    
    maska = wystepy["age"].notna()
    czastkowe = wystepy.loc[maska, ["player_id", "age", "date"]].copy()

    
    czesci = czastkowe["age"].str.split("-", expand=True)
    lata = czesci[0].astype(int)
    dni = czesci[1].astype(int)

    # Krok 1: cofamy sie o liczbe dni od ostatnich urodzin

    rocznica = czastkowe["date"] - pd.to_timedelta(dni, unit="D")

    # Krok 2: cofamy sie o pelne lata. 

    czastkowe["data_urodzenia"] = pd.to_datetime(
        {
            "year": rocznica.dt.year - lata,
            "month": rocznica.dt.month,
            "day": rocznica.dt.day,
        },
        errors="coerce",
    )
    czastkowe = czastkowe.dropna(subset=["data_urodzenia"])

    # Kontrola spojnosci: ta sama data powinna wyjsc z kazdego meczu zawodnika.

    grupy = czastkowe.groupby("player_id")["data_urodzenia"]
    rozrzut = (grupy.max() - grupy.min()).dt.days
    niespojni = int((rozrzut > 0).sum())
    print(
        f"  data urodzenia: odtworzona dla {len(rozrzut)} zawodnikow, "
        f"{niespojni} ma niespojny wiek miedzy meczami "
        f"({100 * niespojni / len(rozrzut):.2f}%, maks rozrzut {rozrzut.max()} dni)"
    )
    print("                  bierzemy date wynikajaca z wiekszosci meczow")

    return wybierz_najczestsza(czastkowe, "player_id", "data_urodzenia")


def kod_na_region(kod) -> str | None:
    """
    Mapuje trzyliterowy kod kraju na jeden z trzech regionow.

    Przyjmuje:
        kod

    Zwraca:
        "AMERYKA_PLD", "EUROPA", "RESZTA"
    """
    
    if pd.isna(kod):
        return None
    if kod in KODY_AMERYKA_PLD:
        return "AMERYKA_PLD"
    if kod in KODY_EUROPA:
        return "EUROPA"
    return "RESZTA"


def etap_3_parsuj(wystepy: pd.DataFrame) -> pd.DataFrame:
    """
    ETAP 3: zamienia trzy kolumny tekstowe FBref na cechy nadajace sie do modelu
    i odsiewa wiersze, ktorych nie da sie zinterpretowac.

    Przyjmuje:
        wystepy - tabela jeden wiersz = jeden wystep w meczu

    Zwraca:
        te sama tabele, wezsza o odsiane wiersze i szersza o nowe kolumny.
    """
    print("\n" + "=" * 70)
    print("ETAP 3  parsowanie: wiek, narodowosc, pozycja")
    print("=" * 70)

    wystepy = wystepy.copy()
    na_wejsciu = len(wystepy)

    wystepy["date"] = pd.to_datetime(wystepy["date"])

    
    # 3a. Wystepy bez minut - odrzucamy, zainfekuja statystyki
    
    
    bez_minut = wystepy["minutes"].isna()
    print(f"\n  odrzucam {bez_minut.sum()} wystepow bez zapisanych minut")
    
    wystepy = wystepy[~bez_minut]

    # 3b. Wiek

    print("\n-- wiek --")
    
    daty_urodzenia = wylicz_daty_urodzenia(wystepy)

    
    wystepy["data_urodzenia"] = wystepy["player_id"].map(daty_urodzenia)

    bez_daty = wystepy["data_urodzenia"].isna()

    print(
        f"  bez daty urodzenia mimo propagacji: {bez_daty.sum()} wystepow "
        f"({wystepy.loc[bez_daty, 'player_id'].nunique()} zawodnikow) - odrzucam"
    )
    wystepy = wystepy[~bez_daty]

    # Wiek liczymy na 30 czerwca roku konczacego sezon - to umowny koniec sezonu
   
    # str[-4:] bierze cztery ostatnie znaki napisu ("2023-2024" daje "2024").
    
    rok_konca = wystepy["season"].str[-4:].astype(int)

    # Liczymy wiek kalendarzowo

    mial_juz_urodziny = wystepy["data_urodzenia"].dt.month <= 6
    wystepy["wiek"] = (
        rok_konca - wystepy["data_urodzenia"].dt.year - (~mial_juz_urodziny).astype(int)
    )
    print(
        f"  wiek na 30.06 roku konczacego sezon: "
        f"od {wystepy['wiek'].min()} do {wystepy['wiek'].max()} lat, "
        f"mediana {wystepy['wiek'].median():.0f}"
    )

    # 3c. Narodowosc

    print("\n-- narodowosc --")

    # Pole nation wyglada jak "eng ENG"

    kody_z_meczow = wystepy["nation"].str.split().str[-1]
    pomocnicza = pd.DataFrame(
        {"player_id": wystepy["player_id"], "kod": kody_z_meczow}
    ).dropna()
    wystepy["kod_kraju"] = wystepy["player_id"].map(
        wybierz_najczestsza(pomocnicza, "player_id", "kod")
    )

    bez_kraju = wystepy["kod_kraju"].isna()
    print(
        f"  bez narodowosci mimo propagacji: {bez_kraju.sum()} wystepow "
        f"({wystepy.loc[bez_kraju, 'player_id'].nunique()} zawodnikow) - odrzucam"
    )
    wystepy = wystepy[~bez_kraju]

    wystepy["region"] = wystepy["kod_kraju"].map(kod_na_region)
    for nazwa_regionu, ile in wystepy["region"].value_counts().items():
        print(
            f"    {nazwa_regionu:12s} {ile:7d} wystepow "
            f"({100 * ile / len(wystepy):5.1f}%)"
        )

    
    reszta = sorted(wystepy.loc[wystepy["region"] == "RESZTA", "kod_kraju"].unique())
    print(f"\n  do RESZTY trafilo {len(reszta)} kodow krajow:")
    print("   ", " ".join(reszta))

    # 3d. Pozycja

    print("\n-- pozycja --")
    # Pozycja bywa zlozona ("DM,CM"), FBref wymienia najpierw pozycje podstawowa

    pierwsza = wystepy["position"].str.split(",").str[0].str.strip()

    # Klase sezonowa wyznaczy ETAP 4 - wedlug sumy minut.

    wystepy["pozycja_mecz"] = pierwsza.map(MAPA_POZYCJI)

    bramkarze = pierwsza == "GK"
    
    nieznane = wystepy["pozycja_mecz"].isna() & ~bramkarze
    print(f"  odrzucam {bramkarze.sum()} wystepow bramkarzy (pozycja GK)")
    print(
        f"  odrzucam {nieznane.sum()} wystepow o nieznanej pozycji: "
        f"{sorted(pierwsza[nieznane].unique())}"
    )
    wystepy = wystepy[wystepy["pozycja_mecz"].notna()]

   
    # 3e. Sezony

    print("\n-- sezony --")
    
    do_analizy = wystepy["season"].isin(SEZONY)
    odrzucone = wystepy.loc[~do_analizy, "season"].value_counts().sort_index()
    for sezon, ile in odrzucone.items():
        print(f"  odrzucam sezon {sezon}: {ile:7d} wystepow")
    wystepy = wystepy[do_analizy]

    
    # Sprawdzenia i kontrolka
    
    assert wystepy["season"].isin(SEZONY).all(), "zostal sezon spoza listy"
    assert wystepy["pozycja_mecz"].isin(["DEF", "MID", "FOR"]).all(), "obca pozycja"
    assert wystepy["minutes"].notna().all(), "zostal wystep bez minut"
    assert wystepy["wiek"].between(14, 50).all(), "wiek poza sensownym zakresem"

    print("\n-- stan po ETAPIE 3 --")
    print(
        f"  wystepow: {na_wejsciu} -> {len(wystepy)} "
        f"(odpadlo {na_wejsciu - len(wystepy)}, "
        f"{100 * (na_wejsciu - len(wystepy)) / na_wejsciu:.1f}%)"
    )
    print(f"  zawodnikow: {wystepy['player_id'].nunique()}")
    print(f"  par zawodnik-sezon: {len(wystepy.groupby(['player_id', 'season']))}")
    print("  nowe kolumny: data_urodzenia, wiek, kod_kraju, region, pozycja_mecz")

    return wystepy


# ETAP 4  agregacja mecz do zawodnik x sezon


def wybierz_wg_sumy_minut(wystepy: pd.DataFrame, kolumna: str) -> pd.Series:
    """
    Dla kazdej pary zawodnik-sezon wybiera te wartosc kolumny, przy ktorej
    zawodnik spedzil na boisku najwiecej MINUT (nie: najwiecej meczow).

    Sluzy do dwoch rzeczy: ustalenia pozycji sezonowej i ligi sezonowej.

    Przyjmuje:
        wystepy - tabela na poziomie meczu, z kolumna "minutes"
        kolumna - co wybieramy, np. "pozycja_mecz" albo "competition"

    Zwraca:
        Series indeksowana (player_id, season), jedna wartosc na pare.

    """
    # Sumujemy minuty w rozbiciu na wartosci kolumny - dostajemy dla kazdej pary
    # tyle wierszy, ile roznych wartosci wystapilo.
    minuty = (
        wystepy.groupby(KLUCZ_SEZON + [kolumna])["minutes"].sum().reset_index()
    )

    minuty = minuty.sort_values(
        KLUCZ_SEZON + ["minutes", kolumna], ascending=[True, True, False, True]
    )
    najlepsze = minuty.drop_duplicates(subset=KLUCZ_SEZON, keep="first")
    return najlepsze.set_index(KLUCZ_SEZON)[kolumna]


def etap_4_agreguj(wystepy: pd.DataFrame) -> pd.DataFrame:
    """
    ETAP 4: sprowadza dane z poziomu pojedynczego meczu na poziom calego sezonu.


    Przyjmuje:
        wystepy

    Zwraca:
        DataFrame o unikalnym kluczu (player_id, season), z surowymi SUMAMI
        sezonowymi. Przeliczenie na 90 minut i wskazniki procentowe robi ETAP 5.

    """
    print("\n" + "=" * 70)
    print("ETAP 4  agregacja mecz -> zawodnik x sezon")
    print("=" * 70)

    na_wejsciu = len(wystepy)

    # 4a. Ktore kolumny sumujemy

    licznikowe = [
        c
        for c in wystepy.columns
        if c not in KOLUMNY_NIELICZNIKOWE and c not in WSKAZNIKI_PROCENTOWE
    ]

    for procent, (licznik, mianowniki) in WSKAZNIKI_PROCENTOWE.items():
        for potrzebna in [licznik, *mianowniki]:
            assert potrzebna in licznikowe, (
                f"kolumna {potrzebna}, potrzebna do policzenia {procent}, "
                f"nie trafila do sumowania"
            )

    print(f"\n  kolumn do zsumowania:        {len(licznikowe)}")
    print(f"  kolumn procentowych odrzuconych: {len(WSKAZNIKI_PROCENTOWE)} "
          f"(policzymy je w ETAPIE 5 z licznikow i mianownikow)")

    
    # 4b. Sumy statystyk
    
    sezony = wystepy.groupby(KLUCZ_SEZON, as_index=False)[licznikowe].sum()

    # 4c. Minuty, mecze i sklad wyjsciowy

    podsumowanie = wystepy.groupby(KLUCZ_SEZON, as_index=False).agg(
        minuty_sezon=("minutes", "sum"),
        mecze=("match_id", "size"),
        mecze_od_poczatku=("start", "sum"),
    )

    # 4d. Cechy stale zawodnika
    
    stale = wystepy.groupby(KLUCZ_SEZON, as_index=False).agg(
        name=("name", "first"),
        data_urodzenia=("data_urodzenia", "first"),
        wiek=("wiek", "first"),
        kod_kraju=("kod_kraju", "first"),
        region=("region", "first"),
    )

    
    for kolumna in ["data_urodzenia", "wiek", "kod_kraju", "region"]:
        ile_roznych = wystepy.groupby(KLUCZ_SEZON)[kolumna].nunique()
        assert (ile_roznych <= 1).all(), (
            f"{kolumna} nie jest stala w obrebie pary zawodnik-sezon"
        )

    # 4e. Pozycja i liga sezonowa

    print("\n-- pozycja i liga sezonowa (wg sumy minut) --")
    pozycja = wybierz_wg_sumy_minut(wystepy, "pozycja_mecz").rename("pozycja")
    liga = wybierz_wg_sumy_minut(wystepy, "competition").rename("liga")

    
    ile_lig = wystepy.groupby(KLUCZ_SEZON)["competition"].nunique()
    zmienil = (ile_lig > 1).rename("zmienil_lige")
    print(f"  par grajacych w wiecej niz jednej lidze: {zmienil.sum()} "
          f"({100 * zmienil.mean():.1f}%)")

    # 4f. Zlozenie wszystkiego w jedna tabele
    
    sezony = sezony.merge(podsumowanie, on=KLUCZ_SEZON, validate="one_to_one")
    sezony = sezony.merge(stale, on=KLUCZ_SEZON, validate="one_to_one")
    for dodatkowa in [pozycja, liga, zmienil]:
        sezony = sezony.merge(
            dodatkowa.reset_index(), on=KLUCZ_SEZON, validate="one_to_one"
        )

    print(f"\n  par zawodnik-sezon po agregacji: {len(sezony)}")

    # 4g. Prog minutowy

    print(f"\n-- prog minutowy (MIN_MINUT = {MIN_MINUT}) --")
    dosc_minut = sezony["minuty_sezon"] >= MIN_MINUT
    print(f"  odrzucam {(~dosc_minut).sum()} par ponizej progu "
          f"({100 * (~dosc_minut).mean():.1f}%)")
    print("  powod: przy kilkunastu minutach statystyki per 90 sa czystym szumem "
          "- jedno podanie w 10 minut daje 9 podan na 90 minut")
    sezony = sezony[dosc_minut]

    # Sprawdzenia i kontrolka

    assert not sezony.duplicated(subset=KLUCZ_SEZON).any(), (
        "para (player_id, season) nie jest unikalna"
    )
    assert (sezony["minuty_sezon"] >= MIN_MINUT).all(), "zostala para ponizej progu"
    assert sezony["pozycja"].isin(["DEF", "MID", "FOR"]).all(), "obca pozycja"

    assert (sezony["mecze_od_poczatku"] <= sezony["mecze"]).all(), (
        "wiecej wyjsc w pierwszym skladzie niz rozegranych meczow"
    )
    
    assert (sezony["minuty_sezon"] <= 90 * sezony["mecze"]).all(), (
        "suma minut przekracza 90 na mecz"
    )

    print("\n-- stan po ETAPIE 4 --")
    print(f"  wystepow na wejsciu:  {na_wejsciu}")
    print(f"  par zawodnik-sezon:   {len(sezony)}")
    print(f"  zawodnikow:           {sezony['player_id'].nunique()}")
    print(f"  kolumn:               {sezony.shape[1]}")
    print(f"  minuty w sezonie:     mediana {sezony['minuty_sezon'].median():.0f}, "
          f"od {sezony['minuty_sezon'].min():.0f} do {sezony['minuty_sezon'].max():.0f}")
    print("\n  rozklad pozycji:")
    for poz, ile in sezony["pozycja"].value_counts().items():
        print(f"    {poz}  {ile:6d} ({100 * ile / len(sezony):5.1f}%)")
    print("\n  par na sezon:")
    for sezon, ile in sezony["season"].value_counts().sort_index().items():
        print(f"    {sezon}  {ile:6d}")

    return sezony


# ETAP 5  wskazniki procentowe i normalizacja per 90 minut


def policz_wskazniki_procentowe(sezony: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Przyjmuje:
        sezony - tabela z zsumowanymi licznikami i mianownikami

    Zwraca:
        tabela z dolozonymi kolumnami procentowymi

    """
    przyciete = {}

    for nazwa, (licznik, mianowniki) in WSKAZNIKI_PROCENTOWE.items():

        mianownik = sezony[mianowniki].sum(axis=1)

        sezony[nazwa] = 100 * sezony[licznik] / mianownik.where(mianownik > 0)

        poza_zakresem = ((sezony[nazwa] < 0) | (sezony[nazwa] > 100)).sum()
        if poza_zakresem:
            przyciete[nazwa] = int(poza_zakresem)

        sezony[nazwa] = sezony[nazwa].clip(lower=0, upper=100)

    return sezony, przyciete


def etap_5_wskazniki_i_p90(sezony: pd.DataFrame) -> pd.DataFrame:
    """
    Przyjmuje:
        sezony - tabela jeden wiersz = para zawodnik-sezon

    Zwraca:
        te sama tabele, poszerzona o kolumny procentowe i kolumny _p90
    """
    print("\n" + "=" * 70)
    print("ETAP 5  wskazniki procentowe i normalizacja per 90 minut")
    print("=" * 70)

    sezony = sezony.copy()

    # 5a. Wskazniki procentowe

    print("\n-- wskazniki procentowe (suma licznikow / suma mianownikow) --")
    sezony, przyciete = policz_wskazniki_procentowe(sezony)

    for nazwa in WSKAZNIKI_PROCENTOWE:
        braki = sezony[nazwa].isna().sum()
        print(
            f"  {nazwa:40s} mediana {sezony[nazwa].median():5.1f}%   "
            f"NaN: {braki:4d} ({100 * braki / len(sezony):4.1f}%)"
        )

    print("\n  przycinanie do [0, 100] (P14):")
    if przyciete:
        for nazwa, ile in przyciete.items():
            print(f"    {nazwa:40s} przyciete {ile} wierszy")
    else:
        print("    zadna wartosc nie wyszla poza zakres")
    print("    NaN oznacza 'nie probowal ani razu' i zostaje brakiem - "
          "to informacja, nie luka do wypelnienia")

    # 5b. Normalizacja na 90 minut

    print("\n-- normalizacja per 90 minut --")

    mecze_po_90 = sezony["minuty_sezon"] / 90

   
    licznikowe = [
        c
        for c in sezony.columns
        if c not in KOLUMNY_NIELICZNIKOWE and c not in WSKAZNIKI_PROCENTOWE
    ]

    p90 = pd.DataFrame(
        {f"{kolumna}_p90": sezony[kolumna] / mecze_po_90 for kolumna in licznikowe},
        index=sezony.index,
    )
    sezony = pd.concat([sezony, p90], axis=1)

    print(f"  przeliczono {len(licznikowe)} statystyk licznikowych")
    print("  surowe sumy zostaja w tabeli (material kontrolny do pliku posredniego)")

    print("\n  kontrola wartosci na 90 minut:")
    for kolumna, opis in [
        ("goals_p90", "gole"),
        ("xG_p90", "oczekiwane gole"),
        ("total_completed_p90", "celne podania"),
        ("tackles_p90", "odbiory"),
        ("total_touches_p90", "dotkniecia pilki"),
    ]:
        if kolumna in sezony.columns:
            print(
                f"    {opis:18s} mediana {sezony[kolumna].median():7.2f}   "
                f"maks {sezony[kolumna].max():7.2f}"
            )

    for nazwa in WSKAZNIKI_PROCENTOWE:
        obecne = sezony[nazwa].dropna()
        assert obecne.between(0, 100).all(), f"{nazwa} poza zakresem [0, 100]"

    
    kolumny_p90 = [c for c in sezony.columns if c.endswith("_p90")]
    assert np.isfinite(sezony[kolumny_p90].to_numpy()).all(), (
        "w kolumnach _p90 pojawila sie nieskonczonosc - dzielenie przez zero"
    )

    assert sezony["goals_p90"].max() < 5, "nierealny wynik goli na 90 minut"

    print("\n-- stan po ETAPIE 5 --")
    print(f"  par zawodnik-sezon: {len(sezony)}")
    print(f"  kolumn lacznie:     {sezony.shape[1]}")
    print(f"    w tym surowe sumy:      {len(licznikowe)}")
    print(f"    w tym statystyki _p90:  {len(kolumny_p90)}")
    print(f"    w tym wskazniki proc.:  {len(WSKAZNIKI_PROCENTOWE)}")

    return sezony


# ETAP 6  crosswalk do Transfermarktu i cechy statyczne


def znormalizuj_nazwe(seria: pd.Series) -> pd.Series:
    """
    Sprowadza nazwiska do wspolnej postaci, po ktorej da sie laczyc oba zrodla.


    Przyjmuje:
        seria - kolumna z nazwiskami

    Zwraca:
        kolumne znormalizowanych nazw.
    """

    rozlozone = seria.astype(str).map(
        lambda tekst: unicodedata.normalize("NFKD", tekst)
    )

    bez_akcentow = (
        rozlozone.str.encode("ascii", "ignore").str.decode("ascii")
    )

    oczyszczone = (
        bez_akcentow.str.lower().str.replace(r"[^a-z ]", "", regex=True)
    )

    return oczyszczone.str.replace(r"\s+", " ", regex=True).str.strip()


def dopasuj_po_kluczu(
    fbref_wolni: pd.DataFrame,
    tm_wolne: pd.DataFrame,
    klucz_fb: list[str],
    klucz_tm: list[str],
) -> pd.DataFrame:
    """


    Przyjmuje:
        fbref_wolni - zawodnicy FBref jeszcze niedopasowani
        tm_wolne    - rekordy TM jeszcze nieprzypisane (z kolumna tm_player_id)
        klucz_fb    - kolumny klucza po stronie FBref
        klucz_tm    - odpowiadajace im kolumny po stronie TM

    Zwraca:
        DataFrame z kolumnami player_id i tm_player_id.

    """

    fb = fbref_wolni.dropna(subset=klucz_fb)
    tm = tm_wolne.dropna(subset=klucz_tm)


    fb = fb.drop_duplicates(subset=klucz_fb, keep=False)
    tm = tm.drop_duplicates(subset=klucz_tm, keep=False)

    # left_on / right_on pozwalaja laczyc kolumny o roznych nazwach po obu
    # stronach - potrzebne, bo FBref ma "kod_kraju", a TM "country_of_citizenship".
    return fb.merge(
        tm[[*klucz_tm, "tm_player_id"]],
        left_on=klucz_fb,
        right_on=klucz_tm,
        how="inner",
        validate="one_to_one",
    )[["player_id", "tm_player_id"]]


def zbuduj_crosswalk(sezony: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    """
    Buduje mapowanie zawodnik FBref -> zawodnik Transfermarkt, kaskada trzystopniowa.

    Zaden pojedynczy klucz nie wystarcza, ale kazdy zawodzi GDZIE INDZIEJ i przez
    to dobrze sie uzupelniaja:

      1. nazwa + data urodzenia
      2. data urodzenia + kraj
      3. sama nazwa

    Przyjmuje:
        sezony - tabela z ETAPU 5
        tm     - wczytany players.csv

    Zwraca:
        DataFrame z kolumnami player_id (FBref) i tm_player_id.
    """
    
    fbref = (
        sezony[["player_id", "name", "data_urodzenia", "kod_kraju"]]
        .drop_duplicates(subset="player_id")
        .copy()
    )
    fbref["nazwa_norm"] = znormalizuj_nazwe(fbref["name"])

    tm = tm.copy().rename(columns={"player_id": "tm_player_id"})
    tm["nazwa_norm"] = znormalizuj_nazwe(tm["name"])
    tm["data_urodzenia_tm"] = pd.to_datetime(tm["date_of_birth"], errors="coerce")

    print(f"  zawodnikow FBref: {len(fbref)}, rekordow w players.csv: {len(tm)}")

    czesci = []          # kolejne kawalki crosswalku
    uzyci_fb = set()     # zawodnicy FBref juz dopasowani
    uzyte_tm = set()     # rekordy TM juz przypisane

    def wolni():
        """Zwraca to, czego dotychczasowe kroki jeszcze nie zuzyly."""

        return (
            fbref[~fbref["player_id"].isin(uzyci_fb)],
            tm[~tm["tm_player_id"].isin(uzyte_tm)],
        )

    def zapisz(wynik: pd.DataFrame, opis: str):
        """Dopisuje wynik kroku i oznacza zuzyte rekordy."""
        czesci.append(wynik)
        uzyci_fb.update(wynik["player_id"])
        uzyte_tm.update(wynik["tm_player_id"])
        print(f"    dopasowano: {len(wynik)} zawodnikow")

    # KROK 1: nazwa + data urodzenia

    print("\n  KROK 1 - nazwa i data urodzenia (najpewniejszy klucz):")
    fb_wolni, tm_wolne = wolni()
    krok1 = dopasuj_po_kluczu(
        fb_wolni, tm_wolne,
        ["nazwa_norm", "data_urodzenia"],
        ["nazwa_norm", "data_urodzenia_tm"],
    )
    zapisz(krok1, "nazwa+data")

    # KROK 2: data urodzenia + kraj
    
    print("\n  KROK 2 - data urodzenia i kraj (nie oglada sie na nazwe):")
    pewne = krok1.merge(fbref[["player_id", "kod_kraju"]], on="player_id").merge(
        tm[["tm_player_id", "country_of_citizenship"]], on="tm_player_id"
    )
    
    mapa_krajow = (
        pewne.dropna(subset=["kod_kraju", "country_of_citizenship"])
        .groupby("kod_kraju")["country_of_citizenship"]
        .agg(lambda kraje: kraje.mode().iloc[0])
        .to_dict()
    )
    fbref["kraj_tm"] = fbref["kod_kraju"].map(mapa_krajow)
    print(f"    mape kod->kraj wyprowadzono z {len(pewne)} pewnych dopasowan "
          f"({len(mapa_krajow)} kodow)")
    bez_mapy = sorted(set(fbref.loc[fbref["kraj_tm"].isna(), "kod_kraju"].dropna()))
    if bez_mapy:
        print(f"    kodow bez odpowiednika: {len(bez_mapy)} {bez_mapy}")
        print("    (ci zawodnicy przechodza do KROKU 3)")

    fb_wolni, tm_wolne = wolni()
    krok2 = dopasuj_po_kluczu(
        fb_wolni, tm_wolne,
        ["data_urodzenia", "kraj_tm"],
        ["data_urodzenia_tm", "country_of_citizenship"],
    )
    zapisz(krok2, "data+kraj")

    # KROK 3: sama nazwa

    print("\n  KROK 3 - sama nazwa (ostatnia szansa):")
    fb_wolni, tm_wolne = wolni()
    krok3 = dopasuj_po_kluczu(fb_wolni, tm_wolne, ["nazwa_norm"], ["nazwa_norm"])
    zapisz(krok3, "nazwa")

    crosswalk = pd.concat(czesci, ignore_index=True)

    assert not crosswalk["player_id"].duplicated().any(), (
        "zawodnik FBref dopasowany dwukrotnie"
    )
    assert not crosswalk["tm_player_id"].duplicated().any(), (
        "ten sam rekord TM przypisany dwom zawodnikom FBref"
    )

    print(f"\n  RAZEM dopasowanych: {len(crosswalk)} z {len(fbref)} zawodnikow "
          f"({100 * len(crosswalk) / len(fbref):.1f}%)")
    return crosswalk



def etap_6_crosswalk(sezony: pd.DataFrame) -> pd.DataFrame:
    """
    ETAP 6: laczy zawodnikow FBref z Transfermarktem i dokleja cechy statyczne.

    Przyjmuje:
        sezony - tabela z ETAPU 5

    Zwraca:
        tabele bez par bez dopasowania, poszerzona o tm_player_id, wzrost_cm
        i noge.
    """
    print("\n" + "=" * 70)
    print("ETAP 6  crosswalk do Transfermarktu i cechy statyczne")
    print("=" * 70)

    na_wejsciu = len(sezony)
    przed_crosswalkiem = sezony

    tm = pd.read_csv(SCIEZKA_TM / "players.csv")

    print("\n-- budowa crosswalku po znormalizowanej nazwie --")
    crosswalk = zbuduj_crosswalk(sezony, tm)

    # 6a. Dopiecie identyfikatora TM

    sezony = sezony.merge(crosswalk, on="player_id", how="inner", validate="many_to_one")
    print(f"\n  par zawodnik-sezon: {na_wejsciu} -> {len(sezony)} "
          f"({100 * len(sezony) / na_wejsciu:.1f}% pokrycia)")
    print(f"  odpadlo {na_wejsciu - len(sezony)} par bez dopasowania do TM")

    for kolumna in ["liga", "region"]:
        przed = przed_crosswalkiem[kolumna].value_counts()
        po = sezony[kolumna].value_counts().reindex(przed.index).fillna(0)
        udzial = (100 * (1 - po / przed)).sort_values(ascending=False)
        for kategoria, procent in udzial.items():
            print(f"    {kategoria:18s} odpadlo {procent:5.1f}%")


    # 6b. Cechy statyczne z TM ktorych nie ma w fbrefie

    cechy = tm[["player_id", "height_in_cm", "foot", "date_of_birth"]].rename(
        columns={
            "player_id": "tm_player_id",
            "height_in_cm": "wzrost_cm",
            "foot": "noga",
            "date_of_birth": "data_urodzenia_tm",
        }
    )
    cechy["data_urodzenia_tm"] = pd.to_datetime(
        cechy["data_urodzenia_tm"], errors="coerce"
    )
    sezony = sezony.merge(cechy, on="tm_player_id", how="left", validate="many_to_one")

    braki_wzrost = sezony["wzrost_cm"].isna().sum()
    braki_noga = sezony["noga"].isna().sum()
    print(f"\n-- cechy statyczne --")
    print(f"  wzrost_cm: brak u {braki_wzrost} par "
          f"({100 * braki_wzrost / len(sezony):.1f}%), "
          f"mediana {sezony['wzrost_cm'].median():.0f} cm")
    print(f"  noga:      brak u {braki_noga} par "
          f"({100 * braki_noga / len(sezony):.1f}%)")
    for wartosc, ile in sezony["noga"].value_counts(dropna=False).items():
        print(f"    {str(wartosc):8s} {ile:6d} ({100 * ile / len(sezony):5.1f}%)")
    print("  brakow NIE uzupelniamy - imputacja nalezy do Pipeline'u przy")
    print("  modelowaniu, zeby nie wyciekla miedzy zbiorem treningowym a testowym")

    # 6c. Pomiar jakosci crosswalku

    print("\n-- jakosc crosswalku: data urodzenia FBref vs TM --")
    maska = sezony["data_urodzenia_tm"].notna()
    roznica = (
        sezony.loc[maska, "data_urodzenia"] - sezony.loc[maska, "data_urodzenia_tm"]
    ).dt.days.abs()
    niezgodne = (roznica > 0).sum()
    print(f"  porownano {maska.sum()} par")
    print(f"    identyczna data:  {(roznica == 0).sum():6d} "
          f"({100 * (roznica == 0).mean():.2f}%)")
    print(f"    roznica 1-7 dni:  {((roznica > 0) & (roznica <= 7)).sum():6d}")
    print(f"    roznica > 7 dni:  {(roznica > 7).sum():6d}")
    print(f"  NIEZGODNYCH: {niezgodne} ({100 * niezgodne / maska.sum():.2f}%)")
    print("  -> tyle mniej wiecej wynosi udzial blednych dopasowan crosswalku")

    # Odrzucamy dopasowania, ktore ten test oblaly (inna data urodzenia)
    
    # Porownanie z NaN zawsze daje False, wiec pary, dla ktorych TM nie podaje daty
    # urodzenia, zostaja w zbiorze - nie mamy podstaw, zeby je odrzucic.

    bledne = (roznica > MAX_ROZNICA_DAT_DNI).reindex(sezony.index, fill_value=False)
    print(f"\n  odrzucam {bledne.sum()} par o rozbieznosci > {MAX_ROZNICA_DAT_DNI} dni "
          f"({sezony.loc[bledne, 'player_id'].nunique()} zawodnikow)")
    print(f"  zostaja pary o rozbieznosci 1-{MAX_ROZNICA_DAT_DNI} dni - tam to niemal "
          f"na pewno ten sam zawodnik")
    sezony = sezony[~bledne]

    # Sprawdzenia

    assert not sezony.duplicated(subset=KLUCZ_SEZON).any(), (
        "para (player_id, season) przestala byc unikalna po crosswalku"
    )
    
    przypisania = sezony.drop_duplicates(subset="player_id")
    assert not przypisania["tm_player_id"].duplicated().any(), (
        "ten sam zawodnik TM przypisany do dwoch roznych zawodnikow FBref"
    )
    
    obecne = sezony["wzrost_cm"].dropna()
    assert obecne.between(150, 215).all(), (
        f"nierealny wzrost w zbiorze: od {obecne.min()} do {obecne.max()} cm"
    )

    print("\n-- stan po ETAPIE 6 --")
    print(f"  par zawodnik-sezon: {len(sezony)}")
    print(f"  zawodnikow:         {sezony['player_id'].nunique()}")
    print(f"  kolumn:             {sezony.shape[1]}")
    print("  nowe kolumny: tm_player_id, wzrost_cm, noga, data_urodzenia_tm")

    return sezony


# ETAP 7  dolaczenie wyceny


def etap_7_dolacz_wycene(sezony: pd.DataFrame) -> pd.DataFrame:
    """
    ETAP 7: dokleja zmienna objasniana - wartosc rynkowa z Transfermarktu.

    Model ma byc modelem wyceny biezacej, dlatego cel to wycena z okolic 30 czerwca,
    czyli z akonczenia sezonu. TM aktualizuje wyceny kilka razy w roku, wiec wybieramy date
    przy danym zawodniku najblizej dnia 30 czerwca.

    Przyjmuje:
        sezony - tabela z ETAPU 6, z kolumna tm_player_id

    Zwraca:
        tabele bez par bez wyceny, poszerzona o market_value_in_eur oraz
        date wyceny i jej odleglosc od 30 czerwca.
    """
    print("\n" + "=" * 70)
    print("ETAP 7  dolaczenie wyceny")
    print("=" * 70)

    na_wejsciu = len(sezony)
    sezony = sezony.copy()

    wyceny = pd.read_csv(SCIEZKA_TM / "player_valuations.csv")
    wyceny["date"] = pd.to_datetime(wyceny["date"])
    wyceny = wyceny.rename(columns={"player_id": "tm_player_id"})
    print(f"\n  wczytano {len(wyceny)} wycen dla {wyceny['tm_player_id'].nunique()} "
          f"zawodnikow TM")

    wyceny = wyceny[wyceny["tm_player_id"].isin(sezony["tm_player_id"])]
    print(f"  po zawezeniu do naszych zawodnikow: {len(wyceny)} wycen")

    
    sezony["rok_t"] = sezony["season"].str[-4:].astype(int)

    # 7a. Wszystkie wyceny kazdej pary

    kandydaci = sezony[["tm_player_id", "season", "rok_t"]].merge(
        wyceny[["tm_player_id", "date", "market_value_in_eur"]],
        on="tm_player_id",
        how="inner",
    )

    # 7b. Okno czasowe

    rok_tekst = kandydaci["rok_t"].astype(str)
    poczatek = pd.to_datetime(rok_tekst + OKNO_WYCENY_OD)
    koniec = pd.to_datetime(rok_tekst + OKNO_WYCENY_DO)
    odniesienie = pd.to_datetime(rok_tekst + DZIEN_ODNIESIENIA)

    w_oknie = kandydaci[
        (kandydaci["date"] >= poczatek) & (kandydaci["date"] <= koniec)
    ].copy()
    w_oknie["dni_od_30_06"] = (w_oknie["date"] - odniesienie[w_oknie.index]).dt.days

    print(f"\n-- wybor wyceny --")
    print(f"  wycen w oknie {OKNO_WYCENY_OD} - {OKNO_WYCENY_DO}: {len(w_oknie)}")

    # 7c. Najblizsza 30 czerwca

    w_oknie["odleglosc"] = w_oknie["dni_od_30_06"].abs()

    w_oknie = w_oknie.sort_values(
        ["tm_player_id", "season", "odleglosc", "date"],
        ascending=[True, True, True, True],
    )
    wybrane = w_oknie.drop_duplicates(subset=["tm_player_id", "season"], keep="first")

    print(f"  par z wycena w oknie: {len(wybrane)} z {na_wejsciu} "
          f"({100 * len(wybrane) / na_wejsciu:.1f}%)")
    print(f"  odleglosc od 30.06: mediana {wybrane['odleglosc'].median():.0f} dni, "
          f"maks {wybrane['odleglosc'].max()}")
    przed = (wybrane["dni_od_30_06"] < 0).sum()
    print(f"    wycen sprzed 30.06: {przed} ({100 * przed / len(wybrane):.1f}%), "
          f"po 30.06: {len(wybrane) - przed}")

    # 7d. Dopiecie do zbioru

    sezony = sezony.merge(
        wybrane[["tm_player_id", "season", "market_value_in_eur", "date",
                 "dni_od_30_06"]].rename(columns={"date": "data_wyceny"}),
        on=["tm_player_id", "season"],
        how="inner",
        validate="one_to_one",
    )
    print(f"\n  odpadlo {na_wejsciu - len(sezony)} par bez wyceny w oknie")

    
    niedodatnie = (sezony["market_value_in_eur"] <= 0).sum()
    if niedodatnie:
        print(f"  odrzucam {niedodatnie} par z wycena <= 0")
        sezony = sezony[sezony["market_value_in_eur"] > 0]

    # Sprawdzenia

    rok_wyceny = sezony["data_wyceny"].dt.year
    assert (rok_wyceny == sezony["rok_t"]).all(), (
        "wycena pochodzi z innego roku niz rok konczacy sezon"
    )
    miesiac = sezony["data_wyceny"].dt.month
    assert miesiac.between(4, 9).all(), (
        f"wycena spoza okna IV-IX: miesiace od {miesiac.min()} do {miesiac.max()}"
    )
    assert not sezony.duplicated(subset=KLUCZ_SEZON).any(), (
        "para (player_id, season) przestala byc unikalna po dolaczeniu wyceny"
    )
    assert (sezony["market_value_in_eur"] > 0).all(), "zostala wycena niedodatnia"

    print("\n-- stan po ETAPIE 7 --")
    print(f"  par zawodnik-sezon: {na_wejsciu} -> {len(sezony)}")
    print(f"  zawodnikow:         {sezony['player_id'].nunique()}")
    print(f"  kolumn:             {sezony.shape[1]}")
    print(f"\n  wartosc rynkowa (EUR):")
    print(f"    mediana  {sezony['market_value_in_eur'].median():>14,.0f}")
    print(f"    srednia  {sezony['market_value_in_eur'].mean():>14,.0f}")
    print(f"    min      {sezony['market_value_in_eur'].min():>14,.0f}")
    print(f"    maks     {sezony['market_value_in_eur'].max():>14,.0f}")
    print("    (srednia mocno powyzej mediany - rozklad jest silnie prawoskosny,")
    print("     stad logarytmowanie celu w ETAPIE 9)")
    
    print("\n  wycena wg sezonu:")
    print(f"    {'sezon':12s} {'mediana':>12s} {'srednia':>12s} {'sr. log':>8s}")
    for sezon, grupa in sezony.groupby("season"):
        print(f"    {sezon:12s} {grupa['market_value_in_eur'].median():>12,.0f} "
              f"{grupa['market_value_in_eur'].mean():>12,.0f} "
              f"{np.log(grupa['market_value_in_eur']).mean():>8.3f}")
    srednie_log = sezony.groupby("season")["market_value_in_eur"].apply(
        lambda x: np.log(x).mean()
    )
    wzrost = 100 * (np.exp(srednie_log.iloc[-1] - srednie_log.iloc[0]) - 1)
    print(f"    wzrost sredniej log przez szesc sezonow: {wzrost:+.1f}%")

    return sezony


# ETAP 8  podzial na trening / kalibracja / test / zbior_C (portugalska)


def etap_8_podziel(sezony: pd.DataFrame) -> pd.DataFrame:
    """
    ETAP 8: przypisuje kazdej parze zawodnik-sezon jeden z czterech zbiorow.

    Cztery zbiory:
      trening     - cztery najstarsze sezony Big 5
      kalibracja  - 2022-2023 Big 5
      test        - 2023-2024 Big 5
      zbior_C     - cala Primeira Liga, wszystkie sezony

    zbior_C sluzy do pomiaru przesuniecia dziedziny

    Przyjmuje:
        sezony - tabela z ETAPU 7

    Zwraca:
        te sama tabele z dolozona kolumna "podzial".
    """
    print("\n" + "=" * 70)
    print("ETAP 8  podzial na trening / kalibracja / test / zbior_C")
    print("=" * 70)

    sezony = sezony.copy()

    warunki = [
        sezony["liga"] == LIGA_ZBIOR_C,
        sezony["season"].isin(SEZONY_TRENING),
        sezony["season"].isin(SEZONY_KALIBRACJA),
        sezony["season"].isin(SEZONY_TEST),
    ]
    nazwy = ["zbior_C", "trening", "kalibracja", "test"]
    
    sezony["podzial"] = np.select(warunki, nazwy, default="")

    print()
    for nazwa in nazwy:
        pod = sezony[sezony["podzial"] == nazwa]
        sezony_w_zbiorze = ", ".join(sorted(pod["season"].unique()))
        print(f"  {nazwa:12s} {len(pod):6d} par ({100 * len(pod) / len(sezony):5.1f}%)  "
              f"{pod['player_id'].nunique():5d} zawodnikow")
        print(f"  {'':12s} sezony: {sezony_w_zbiorze}")
        print(f"  {'':12s} ligi:   {', '.join(sorted(pod['liga'].unique()))}")

    # Sprawdzenia

    assert (sezony["podzial"] != "").all(), "sa pary bez przypisanego zbioru"

    for zbior in ["trening", "kalibracja"]:
        ligi = set(sezony.loc[sezony["podzial"] == zbior, "liga"])
        assert LIGA_ZBIOR_C not in ligi, (
            f"{LIGA_ZBIOR_C} pojawila sie w zbiorze {zbior} - eksperyment "
            f"z przesunieciem dziedziny nic nie mierzy"
        )

    
    assert not sezony.duplicated(subset=KLUCZ_SEZON).any(), (
        "ta sama para zawodnik-sezon wystepuje w wiecej niz jednym zbiorze"
    )

    
    w_treningu = set(sezony.loc[sezony["podzial"] == "trening", "player_id"])
    w_c = set(sezony.loc[sezony["podzial"] == "zbior_C", "player_id"])
    wspolni = w_treningu & w_c
    print(f"\n  UWAGA: {len(wspolni)} zawodnikow wystepuje i w treningu, i w zbiorze C")
    print("    (w roznych sezonach - pary sa rozlaczne, wiec to nie wyciek,")
    print("     ale model zna juz tych zawodnikow, co zanizy pomiar W2)")
    pary_c = sezony[(sezony["podzial"] == "zbior_C")
                    & sezony["player_id"].isin(wspolni)]
    print(f"    dotyczy {len(pary_c)} z "
          f"{(sezony['podzial'] == 'zbior_C').sum()} par zbioru C "
          f"({100 * len(pary_c) / (sezony['podzial'] == 'zbior_C').sum():.1f}%)")

    return sezony


# ETAP 9  deflacja (indeks liczony tylko na treningu)


def policz_indeks_inflacji(trening: pd.DataFrame) -> dict[str, float]:
    """
    Przyjmuje:
        trening - WYLACZNIE wiersze ze zbioru treningowego

    Zwraca:
        slownik {sezon: wspolczynnik}, gdzie pierwszy sezon ma 1.0.

    Zaklada:
        ze komorki liczace mniej niz MIN_OBSERWACJI_KOMORKA obserwacji sa zbyt
        male, zeby mediana byla stabilna - takie pomijamy.
    """
    dane = trening.copy()
    dane["log_wartosc"] = np.log(dane["market_value_in_eur"])
    # pd.cut dzieli liczby na przedzialy. right=False znaczy, ze przedzial
    # domyka sie od lewej: [24, 26) to wiek 24 i 25, ale juz nie 26.
    dane["komorka_wieku"] = pd.cut(dane["wiek"], PRZEDZIALY_WIEKU, right=False)

    sezony_rosnaco = sorted(dane["season"].unique())
    wspolczynniki = {sezony_rosnaco[0]: 1.0}
    biezacy = 1.0

    print(f"  {'para sezonow':28s} {'zmiana':>9s} {'komorek':>9s} {'obs.':>7s}")
    for wczesniejszy, pozniejszy in zip(sezony_rosnaco, sezony_rosnaco[1:]):
        grupa_a = dane[dane["season"] == wczesniejszy]
        grupa_b = dane[dane["season"] == pozniejszy]

        roznice, wagi = [], []
        for komorka in dane["komorka_wieku"].cat.categories:
            for pozycja in ["DEF", "MID", "FOR"]:
                a = grupa_a[(grupa_a["komorka_wieku"] == komorka)
                            & (grupa_a["pozycja"] == pozycja)]["log_wartosc"]
                b = grupa_b[(grupa_b["komorka_wieku"] == komorka)
                            & (grupa_b["pozycja"] == pozycja)]["log_wartosc"]
                if len(a) >= MIN_OBSERWACJI_KOMORKA and len(b) >= MIN_OBSERWACJI_KOMORKA:
                    # Roznica srednich logarytmow to logarytm ilorazu srednich
                    # geometrycznych - czyli procentowa zmiana ceny wewnatrz tej
                    # samej komorki.
                    #
                    # UWAGA - tu odchodzimy od litery P16, ktora mowi o MEDIANIE.
                    # Powod jest zmierzony: Transfermarkt wycenia w okraglych
                    # progach i cala zmienna przyjmuje tylko 123 rozne wartosci
                    # w calym zbiorze. Mediana komorki liczacej 60-140 obserwacji
                    # nie jest wtedy wielkoscia ciagla - skacze miedzy sasiednimi
                    # progami (5 mln -> 8 mln), co daje "inflacje" +60% albo
                    # +200% tam, gdzie rynek ledwie drgnal. Zmierzone: mediana
                    # dawala +40,0% dla pary 2017/18 -> 2018/19, srednia +31,9%,
                    # a surowa zmiana poziomu w Big 5 wynosi +30,0% - czyli to
                    # srednia trzyma sie danych, a mediana od nich odlatuje.
                    #
                    # Odpornosc mediany na wartosci skrajne jest tu mniej
                    # potrzebna, bo pracujemy juz na logarytmach, ktore same
                    # tlumia dlugi ogon rozkladu.
                    roznice.append(b.mean() - a.mean())
                    wagi.append(len(a))

        # np.average z wagami: komorki liczniejsze wazą wiecej, bo ich mediana
        # jest pewniejsza. Wagi bierzemy z sezonu WCZESNIEJSZEGO - to on jest
        # baza porownania.
        zmiana = np.average(roznice, weights=wagi) if roznice else 0.0
        biezacy *= np.exp(zmiana)
        wspolczynniki[pozniejszy] = biezacy
        print(f"  {wczesniejszy} -> {pozniejszy}   {100 * (np.exp(zmiana) - 1):>+8.1f}% "
              f"{len(roznice):>9d} {sum(wagi):>7d}")

    return wspolczynniki


def etap_9_deflacja(sezony: pd.DataFrame) -> pd.DataFrame:
    """
    ETAP 9: dokleja cel zdeflowany oraz logarytmy obu wersji celu.

    WAZNE - co jest tu celem glownym. Inflacja rynku jest REALNA i duza: ceny
    w Big 5 rosna o okolo 38% miedzy sezonem 2017-2018 a 2023-2024, a poziom
    zbioru testowego lezy o 18,4% wyzej niz treningowego.
    (Uwaga interpretacyjna: mierzac to na calym zbiorze razem z Primeira Liga
    dostaje sie okolo 5%, ale to artefakt - Primeira Liga wchodzi do danych
    dopiero od sezonu 2018-2019 i ma nizsze wyceny, wiec jej pojawienie sie
    maskuje wzrost cen. Wlasnie przed tym chroni staly sklad porownania.)

    Mimo to celem GLOWNYM jest log_market_value (NOMINALNY), a wersja realna
    zostaje jako kolumna dodatkowa. Powod: test na modelu (HistGradientBoosting,
    trening -> test) pokazal, ze deflacja prawie nic nie zmienia:
        bez deflacji   MAE 0,5734   R2 0,6487   obciazenie -6,5%
        z deflacja     MAE 0,5757   R2 0,6496   obciazenie -5,9%
    Roznica poziomow +18,4% przeklada sie na obciazenie zaledwie -6,5%, bo
    drzewa nie ekstrapoluja poza zakres widziany w treningu. Deflacja zostaje
    wiec policzona i zapisana jako material do analizy wrazliwosci, ale nie
    komplikuje glownego toru.

    Przyjmuje:
        sezony - tabela z ETAPU 8, z kolumna "podzial"

    Zwraca:
        tabele z kolumnami market_value_real, log_market_value,
        log_market_value_real.
    """
    print("\n" + "=" * 70)
    print("ETAP 9  deflacja (indeks liczony tylko na treningu)")
    print("=" * 70)

    sezony = sezony.copy()

    # Do funkcji liczacej indeks przekazujemy WYLACZNIE zbior treningowy i tylko
    # Big 5. To realizacja sprawdzenia nr 7: funkcja fizycznie nie widzi danych
    # kalibracyjnych, testowych ani portugalskich.
    trening = sezony[sezony["podzial"] == "trening"]
    print(f"\n  indeks liczony na {len(trening)} parach ze zbioru treningowego")
    print("  (funkcja nie dostaje danych z kalibracji, testu ani zbioru C)")
    print()
    wspolczynniki = policz_indeks_inflacji(trening)

    # Sezony spoza treningu dostaja wspolczynnik ostatniego sezonu treningowego -
    # "przedluzenie plaskie". Nie wolno policzyc ich wlasnych wspolczynnikow, bo
    # wymagaloby to zajrzenia do wycen ze zbioru testowego.
    ostatni = wspolczynniki[max(wspolczynniki)]
    for sezon in sezony["season"].unique():
        if sezon not in wspolczynniki:
            wspolczynniki[sezon] = ostatni

    print(f"\n  wspolczynniki indeksu:")
    for sezon in sorted(wspolczynniki):
        zrodlo = "trening" if sezon in set(trening["season"]) else "przedluzenie plaskie"
        print(f"    {sezon}  {wspolczynniki[sezon]:.4f}   ({zrodlo})")

    # ------------------------------------------------------------------
    # Cele
    # ------------------------------------------------------------------
    sezony["market_value_real"] = (
        sezony["market_value_in_eur"] / sezony["season"].map(wspolczynniki)
    )

    # log1p to log(1 + x). Przy wycenach rzedu milionow rozni sie od zwyklego
    # logarytmu w dwudziestym miejscu po przecinku, ale jest odporny na zero,
    # gdyby kiedys jakies przeszlo przez filtry ETAPU 7.
    sezony["log_market_value"] = np.log1p(sezony["market_value_in_eur"])
    sezony["log_market_value_real"] = np.log1p(sezony["market_value_real"])

    print(f"\n  CEL GLOWNY:    log_market_value (nominalny)")
    print("  cel dodatkowy: log_market_value_real (deflowany, do analizy wrazliwosci)")

    assert sezony["market_value_real"].gt(0).all(), "niedodatnia wartosc realna"
    assert sezony["log_market_value"].notna().all(), "brak w celu glownym"

    print(f"\n  skosnosc celu: surowo {sezony['market_value_in_eur'].skew():.2f}, "
          f"po log {sezony['log_market_value'].skew():.2f}")

    return sezony


# ETAP 10  zapis plikow


def etap_10_zapisz(sezony: pd.DataFrame) -> pd.DataFrame:
    """
    ETAP 10: zapisuje dwa pliki wynikowe do data/processed/.

    Powstaja dwa zbiory o roznym przeznaczeniu:

      zawodnik_sezon.parquet - WSZYSTKO, lacznie z surowymi sumami sezonowymi
          i kolumnami pomocniczymi. Material kontrolny: gdy cos w wynikach
          modelu wyglada podejrzanie, tu mozna sprawdzic, skad sie wzielo.

      model.parquet - tylko to, co wchodzi do modelu: cechy, cele, klucz
          i kolumna "podzial". Waski zbior zmniejsza ryzyko, ze do modelu
          trafi przypadkiem kolumna, ktora trafic nie powinna.

    Dlaczego parquet, a nie CSV: przy ponad dwustu kolumnach CSV gubi typy -
    liczby wracaja jako tekst, daty jako napisy, a braki jako puste ciagi
    nieodroznialne od zera. Parquet zapisuje typy razem z danymi.

    Przyjmuje:
        sezony - tabela z ETAPU 9

    Zwraca:
        zbior modelowy (ten sam, ktory zapisuje do model.parquet).
    """
    print("\n" + "=" * 70)
    print("ETAP 10  zapis plikow")
    print("=" * 70)

    # mkdir tworzy katalog. parents=True tworzy tez katalogi nadrzedne, jesli
    # ich nie ma, a exist_ok=True sprawia, ze istniejacy katalog nie jest bledem.
    KATALOG_WYJSCIA.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 10a. Cecha pochodna: kwadrat wieku
    # ------------------------------------------------------------------
    # Zaleznosc wartosci od wieku nie jest liniowa - zawodnik zyskuje na wartosci
    # mniej wiecej do 25 roku zycia, a potem traci. Model liniowy z samym "wiek"
    # moglby uchwycic tylko monotoniczny trend, wiec dokladamy kwadrat, ktory
    # pozwala opisac odwrocona litere U. Modele drzewiaste poradza sobie i bez
    # tego, ale cecha nie szkodzi, a porownanie rodzin modeli staje sie uczciwsze.
    sezony = sezony.copy()
    sezony["wiek_do_kw"] = sezony["wiek"] ** 2

    # ------------------------------------------------------------------
    # 10b. Pelny zbior kontrolny
    # ------------------------------------------------------------------
    sciezka_pelny = KATALOG_WYJSCIA / "zawodnik_sezon.parquet"
    sezony.to_parquet(sciezka_pelny, index=False)
    print(f"\n  zawodnik_sezon.parquet  {len(sezony):6d} wierszy x "
          f"{sezony.shape[1]:3d} kolumn  "
          f"{sciezka_pelny.stat().st_size / 1024 / 1024:.1f} MB")

    # ------------------------------------------------------------------
    # 10c. Zbior modelowy
    # ------------------------------------------------------------------
    # Cechy licznikowe wchodza WYLACZNIE w wersji _p90. Surowe sumy sezonowe
    # zostaja w pliku kontrolnym, bo mieszaja jakosc zawodnika z liczba minut:
    # zawodnik z 10 golami w 3400 minut jest gorszym strzelcem niz ten z 8 golami
    # w 900 minut, a suma mowi odwrotnie.
    cechy_p90 = sorted(c for c in sezony.columns if c.endswith("_p90"))
    cechy_procentowe = sorted(WSKAZNIKI_PROCENTOWE)

    kolumny_modelu = [
        # klucz - nie cecha, ale bez niego nie da sie niczego sprawdzic
        *KLUCZ_SEZON,
        # metadane: do identyfikacji wiersza przy analizie bledow, NIE do modelu
        "name", "tm_player_id",
        # przynaleznosc do zbioru
        "podzial",
        # cele
        "market_value_in_eur", "market_value_real",
        "log_market_value", "log_market_value_real",
        # cechy liczbowe
        *cechy_p90,
        *cechy_procentowe,
        "wiek", "wiek_do_kw", "wzrost_cm", "minuty_sezon", "mecze",
        # cechy kategoryczne - zostawiamy jako TEKST. Kodowanie ma sie dziac
        # wewnatrz Pipeline'u przy modelowaniu, zeby kategorie nie rozjechaly sie
        # miedzy zbiorem treningowym a testowym (np. gdy jakas liga wystapi tylko
        # w jednym z nich).
        "pozycja", "region", "liga", "noga", "zmienil_lige",
    ]

    brakujace = [c for c in kolumny_modelu if c not in sezony.columns]
    assert not brakujace, f"brak kolumn w zbiorze: {brakujace}"

    model = sezony[kolumny_modelu]
    sciezka_model = KATALOG_WYJSCIA / "model.parquet"
    model.to_parquet(sciezka_model, index=False)
    print(f"  model.parquet           {len(model):6d} wierszy x "
          f"{model.shape[1]:3d} kolumn  "
          f"{sciezka_model.stat().st_size / 1024 / 1024:.1f} MB")

    # ------------------------------------------------------------------
    # Czego swiadomie NIE ma w zbiorze modelowym
    # ------------------------------------------------------------------
    print("\n  swiadomie pominiete w zbiorze modelowym:")
    print("    surowe sumy sezonowe   - kazda ma odpowiednik _p90")
    print("    market_value_prev      - zeszloroczna wycena wyjasnia 83,75%")
    print("                             wariancji celu i zdominowalaby SHAP,")
    print("                             czyniac wymiar W4 bezprzedmiotowym")
    print("    attendance             - 94,2% brakow w sezonie 2020-2021,")
    print("                             braki skorelowane z sezonem")
    print("    data_urodzenia, age    - zastapione przez wiek")

    # ------------------------------------------------------------------
    # Sprawdzenia koncowe
    # ------------------------------------------------------------------
    assert not model.duplicated(subset=KLUCZ_SEZON).any(), "klucz nie jest unikalny"
    assert model["log_market_value"].notna().all(), "brak w celu glownym"
    assert (model["minuty_sezon"] >= MIN_MINUT).all(), "para ponizej progu minut"
    for zbior in ["trening", "kalibracja"]:
        assert LIGA_ZBIOR_C not in set(model.loc[model["podzial"] == zbior, "liga"]), (
            f"{LIGA_ZBIOR_C} w zbiorze {zbior}"
        )
    for kolumna in cechy_procentowe:
        obecne = model[kolumna].dropna()
        assert obecne.between(0, 100).all(), f"{kolumna} poza [0, 100]"

    print("\n-- ZBIOR MODELOWY --")
    print(f"  wierszy: {len(model)}")
    print(f"  cech liczbowych:     {len(cechy_p90) + len(cechy_procentowe) + 5}")
    print(f"    w tym _p90:        {len(cechy_p90)}")
    print(f"    w tym procentowe:  {len(cechy_procentowe)}")
    print("  cech kategorycznych: 5  (pozycja, region, liga, noga, zmienil_lige)")
    print("  celow:               4  (nominalny i realny, kazdy surowy i log)")
    print(f"\n  podzial:")
    for nazwa, ile in model["podzial"].value_counts().items():
        print(f"    {nazwa:12s} {ile:6d} ({100 * ile / len(model):5.1f}%)")

    print(f"\n  braki w cechach (kolumny, gdzie cokolwiek brakuje):")
    braki = model.isna().sum()
    braki = braki[braki > 0].sort_values(ascending=False)
    if len(braki):
        for kolumna, ile in braki.items():
            print(f"    {kolumna:42s} {ile:5d} ({100 * ile / len(model):4.1f}%)")
        print("    (braki strukturalne - 'nie probowal ani razu' - zostaja jako NaN;")
        print("     imputacja nalezy do Pipeline'u przy modelowaniu)")
    else:
        print("    brak")

    return model


if __name__ == "__main__":
    tabele = etap_1_wczytaj_i_zdeduplikuj(SCIEZKA_DB)
    wystepy = etap_2_polacz_tabele(tabele)
    wystepy = etap_3_parsuj(wystepy)
    sezony = etap_4_agreguj(wystepy)
    sezony = etap_5_wskazniki_i_p90(sezony)
    sezony = etap_6_crosswalk(sezony)
    sezony = etap_7_dolacz_wycene(sezony)
    sezony = etap_8_podziel(sezony)
    sezony = etap_9_deflacja(sezony)
    model = etap_10_zapisz(sezony)
