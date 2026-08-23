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
from pathlib import Path

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


# Wskazniki procentowe

# NAJWAZNIEJSZA REGULA AGREGACJI - procentow NIGDY nie usredniamy z poziomu
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
    print(f"  powod: przy kilkunastu minutach statystyki per 90 sa czystym szumem "
          f"- jedno podanie w 10 minut daje 9 podan na 90 minut")
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


if __name__ == "__main__":
    tabele = etap_1_wczytaj_i_zdeduplikuj(SCIEZKA_DB)
    wystepy = etap_2_polacz_tabele(tabele)
    wystepy = etap_3_parsuj(wystepy)
    sezony = etap_4_agreguj(wystepy)
