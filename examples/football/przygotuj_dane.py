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

# ---------------------------------------------------------------------
# ETAP 3: sezony, pozycje, regiony
# ---------------------------------------------------------------------

# Sezony wchodzace do analizy. Z dziewieciu dostepnych zostaje szesc:
#   2019-2020 odpada - Ligue 1 rozegrala 279 z 380 meczow (pandemia przerwala
#                      rozgrywki), wiec sumy sezonowe Francuzow sa nieporownywalne
#   2024-2025 odpada - 25% brakow zmiennej celu, skorelowanych z liga
#   2025-2026 odpada - sezon w trakcie, dane niekompletne z definicji
SEZONY = [
    "2017-2018",
    "2018-2019",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
]

# Mapowanie pozycji FBref na trzy klasy. FBref rozroznia kilkanascie pozycji;
# my sprowadzamy je do formacji, bo przy trzech klasach kazda ma dosc obserwacji,
# zeby model nauczyl sie jej specyfiki (napastnik wyceniany jest za gole,
# obronca za odbiory).
#
# Nie ma tu GK. Bramkarzy odrzucamy - ocenia sie ich zupelnie innymi metrykami
# (obrony, post-shot xG), ktorych zawodnicy z pola nie maja, wiec wspolny model
# musialby dla nich zgadywac z samych zer.
MAPA_POZYCJI = {
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "WB": "DEF",
    "DM": "MID", "CM": "MID", "LM": "MID", "RM": "MID", "AM": "MID",
    "LW": "FOR", "RW": "FOR", "FW": "FOR",
}

# Kraje CONMEBOL - Ameryka Poludniowa jako osobny region. Wyodrebniamy ja, bo
# rynek transferowy wycenia zawodnikow z Ameryki Poludniowej inaczej: dochodzi
# premia za "potencjal" i koszt adaptacji do Europy. To wlasnie na tej zmiennej
# bedzie sie opierac czesc pomiaru sprawiedliwosci (W5).
#
# Uwaga: `frozenset` zamiast listy. Sprawdzenie "czy kod jest w zbiorze" dziala
# w zbiorze w czasie stalym (haszowanie), a w liscie wymaga przejrzenia jej
# element po elemencie. Przy 500 tysiacach wierszy roznica jest odczuwalna.
# "frozen" znaczy, ze zbioru nie da sie przypadkiem zmodyfikowac po utworzeniu.
KODY_AMERYKA_PLD = frozenset({
    "ARG", "BOL", "BRA", "CHI", "COL", "ECU", "PAR", "PER", "URU", "VEN",
})

# Kraje UEFA - pelna lista 55 federacji, takze tych nieobecnych w naszych danych
# (GIB, SMR, KAZ), zeby lista nie wymagala poprawek przy zmianie zakresu danych.
#
# Uwaga na trzy rzeczy, ktore lamia intuicje geograficzna:
#   - ISR, ARM, AZE, GEO, KAZ, RUS naleza do UEFA, choc lezy to poza Europa
#     w scislym sensie geograficznym. Idziemy za UEFA, bo to podzial rynku
#     pilkarskiego, a nie lekcja geografii - te kraje graja w europejskich
#     pucharach i ich zawodnicy podlegaja tym samym przepisom transferowym.
#   - Wielka Brytania wystepuje jako cztery osobne federacje: ENG, SCO, WAL, NIR.
#   - Kosowo ma w FBref kod KVX (nie KOS) - sprawdzone w danych, 1231 wystepow.
KODY_EUROPA = frozenset({
    "ALB", "AND", "ARM", "AUT", "AZE", "BLR", "BEL", "BIH", "BUL", "CRO",
    "CYP", "CZE", "DEN", "ENG", "EST", "FRO", "FIN", "FRA", "GEO", "GER",
    "GIB", "GRE", "HUN", "ISL", "ISR", "ITA", "KAZ", "KVX", "LVA", "LIE",
    "LTU", "LUX", "MLT", "MDA", "MNE", "NED", "MKD", "NIR", "NOR", "POL",
    "POR", "IRL", "ROU", "RUS", "SMR", "SCO", "SRB", "SVK", "SVN", "ESP",
    "SWE", "SUI", "TUR", "UKR", "WAL",
})



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
    # Budujemy slownik {nazwa_kolumny: [tabele, w ktorych wystepuje]} i wymuszamy,
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


if __name__ == "__main__":
    tabele = etap_1_wczytaj_i_zdeduplikuj(SCIEZKA_DB)
    wystepy = etap_2_polacz_tabele(tabele)
