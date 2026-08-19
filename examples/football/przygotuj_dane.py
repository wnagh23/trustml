"""
Przygotowanie zbioru modelowego: dane FBref (mecz po meczu) + Transfermarkt (wycena)
-> jeden wiersz na pare (zawodnik, sezon).

Ten skrypt NIE modeluje. Konczy sie na zapisaniu plikow do data/processed/.

Semantyka celu: to jest model WYCENY BIEZACEJ, nie prognoza. Cechy opisuja sezon
konczacy sie w roku t, a cel to wycena Transfermarktu z okolic 30 czerwca roku t.
Model odpowiada na pytanie "ile ten zawodnik jest wart teraz, sadzac po tym, jak
zagral ten sezon", a nie "ile bedzie wart za rok".

Uwaga o zapisie: caly plik jest bez polskich znakow diakrytycznych. Powod jest
praktyczny - konsola Windows w tym srodowisku nie wyswietla poprawnie UTF-8,
wiec kontrolki drukowane przez print() rozsypywalyby sie na ekranie. Reszta
dokumentacji projektu (docs/06, docs/07) trzyma te sama konwencje.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

# =====================================================================
# PARAMETRY - wszystko, co moglbys chciec zmienic, jest w tym jednym miejscu
# =====================================================================

# Sciezki. Dane leza w repozytorium, w data/raw/. Katalog data/raw/ jest TYLKO
# DO ODCZYTU - skrypt nigdy tam nic nie zapisuje ani nie modyfikuje.
KATALOG_REPO = Path(__file__).resolve().parents[2]
SCIEZKA_DB = KATALOG_REPO / "data" / "raw" / "fbref" / "master.db"
SCIEZKA_TM = KATALOG_REPO / "data" / "raw" / "tm"
KATALOG_WYJSCIA = KATALOG_REPO / "data" / "processed"

# Tabele statystyk polowych. Goalkeeper celowo pomijamy - odrzucamy bramkarzy,
# wiec ta tabela nie ma dla nas zastosowania.
TABELE_POLOWE = [
    "Summary",
    "Passing",
    "Possession",
    "Defensive_Actions",
    "Pass_Types",
    "Miscellaneous",
]

# Klucz zlozony, po ktorym lacza sie wszystkie tabele FBref.
KLUCZ = ["match_id", "player_id"]

# Co bierzemy z tabeli Match. Tabela ma 15 kolumn, ale opisuja one MECZ, a nie
# zawodnika - wynik, gospodarza, sedziego, stadion. Dla modelu oceniajacego
# pojedynczego zawodnika sa albo nieistotne, albo wrecz szkodliwe, bo opisuja
# skutek gry calej druzyny. Bierzemy tylko to, bez czego nie da sie pracowac:
#   competition - liga, potrzebna do podzialu zbioru (ETAP 8) i jako cecha
#   season      - sezon, czyli jednostka agregacji (ETAP 4)
#   date        - data meczu, bez ktorej nie policzymy daty urodzenia (ETAP 3)
#
# Swiadomie NIE bierzemy attendance. Frekwencja ma 94,2% brakow w sezonie
# 2020-2021 (mecze przy pustych trybunach) i 25,0% w 2019-2020. Braki nie sa
# losowe, tylko skorelowane z sezonem, wiec jako cecha bylaby w praktyce
# ukrytym znacznikiem "to sezon pandemiczny".
KOLUMNY_Z_MATCH = ["match_id", "competition", "season", "date"]

# ---------------------------------------------------------------------
# Redundancja kolumn miedzy tabelami - ustalona pomiarem, nie zalozeniem
# ---------------------------------------------------------------------
# FBref powiela te same metryki w kilku tabelach. Sprawdzilem wartosc po wartosci
# na wszystkich 521 261 wierszach: ponizsze kolumny sa IDENTYCZNE z odpowiednikiem
# w tabeli szczegolowej (zero rozbieznosci). Zostawiam wersje z tabeli
# szczegolowej, bo tam metryka ma pelny kontekst (np. tackles obok tackles_won
# i podzialu na strefy boiska), a kopie usuwam.
#
# Dlaczego nie sufiksowac ich zamiast usuwac (tackles_summary vs tackles_def)?
# Bo model dostalby wtedy 16 par cech skorelowanych w 100%. Dwie idealnie
# skorelowane cechy dziela miedzy siebie waznosc w SHAP, wiec obie wygladaja na
# polowicznie wazne - to psuje wymiar W4 (wyjasnialnosc, mierzona m.in.
# stabilnoscia rankingu waznosci cech). Redundancja szkodzi tez W2 (odpornosc):
# usuniecie jednej cechy z pary nic nie zmienia, bo blizniaczka ja zastepuje,
# wiec test wrazliwosci na usuniecie cechy falszywie pokazuje model jako odporny.
#
# UWAGA - dwa bledy etykiet w tabeli Summary, wykryte przy tym pomiarze:
#
#   1. Summary.successful_dribbles i Summary.dribbles_attempted sa ZAMIENIONE
#      MIEJSCAMI. Dowod: porownanie na krzyz z tabela Possession zgadza sie w
#      521 261 z 521 261 wierszy, a w samym Summary jest 173 012 wierszy, gdzie
#      "udane" przewyzsza "probowane" - to fizycznie niemozliwe. W Possession
#      takich wierszy jest zero.
#
#   2. Summary.xA to w rzeczywistosci xAG (expected assisted goals), a nie xA.
#      Zgadza sie z Passing.xAG w 100% wierszy, a z Passing.xA tylko w 74%.
#
# Obu bledow nie naprawiamy w bazie (data/raw/ jest tylko do odczytu) i nie
# musimy ich mapowac przy odczycie, bo obie felerne kolumny i tak usuwamy jako
# redundantne - dryblingi bierzemy z Possession, a xAG z Passing pod wlasciwa
# nazwa.
KOLUMNY_ZREDUNDOWANE = {
    "Summary": [
        # te same wartosci, ta sama nazwa; tabela zrodlowa w komentarzu
        "assists",                     # -> Passing.assists
        "blocks",                      # -> Defensive_Actions.blocks
        "carries",                     # -> Possession.carries
        "interceptions",               # -> Defensive_Actions.interceptions
        "progressive_carries",         # -> Possession.progressive_carries
        "progressive_passes",          # -> Passing.progressive_passes
        "red_cards",                   # -> Miscellaneous.red_cards
        "tackles",                     # -> Defensive_Actions.tackles
        "yellow_cards",                # -> Miscellaneous.yellow_cards
        # te same wartosci, INNA nazwa w tabeli zrodlowej
        "touches",                     # -> Possession.total_touches
        "passes_completed",            # -> Passing.total_completed
        "passes_attempted",            # -> Passing.total_attempted
        "pass_completion_percentage",  # -> Passing.completion_percentage
        # kolumny z bledna etykieta - patrz UWAGA wyzej
        "successful_dribbles",         # -> Possession.successful_dribbles
        "dribbles_attempted",          # -> Possession.dribbles_attempted
        "xA",                          # -> Passing.xAG
    ],
    "Pass_Types": [
        "passes_completed",            # -> Passing.total_completed
        "passes_attempted",            # -> Passing.total_attempted
    ],
    "Miscellaneous": [
        "crosses",                     # -> Pass_Types.crosses
        "interceptions",               # -> Defensive_Actions.interceptions
        "tackles_won",                 # -> Defensive_Actions.tackles_won
    ],
}

# Kolizja nazw, ktora NIE jest redundancja - to dwie rozne metryki, ktore
# przypadkiem nazywaja sie tak samo, i rozniace sie w 98 900 wierszach:
#   Pass_Types.offsides    - podania zagrane na spalonego (wina podajacego)
#   Miscellaneous.offsides - spalone zlapane przez zawodnika (wina przyjmujacego)
# Obie zostaja, ale ta z Pass_Types dostaje jednoznaczna nazwe.
ZMIANY_NAZW = {
    "Pass_Types": {"offsides": "offsides_z_podan"},
}


# =====================================================================
# ETAP 1  wczytanie i deduplikacja tabel z master.db
# =====================================================================


def wczytaj_tabele(polaczenie: sqlite3.Connection, nazwa: str) -> pd.DataFrame:
    """
    Wczytuje jedna tabele z master.db i usuwa z niej zdublowane wiersze.

    Przyjmuje:
        polaczenie - otwarte polaczenie sqlite3 do master.db
        nazwa      - nazwa tabeli, np. "Summary"

    Zwraca:
        DataFrame o unikalnym kluczu (match_id, player_id).

    Zaklada:
        ze zdublowane klucze pochodza z dokladnych kopii wierszy. Jesli kiedys
        pojawi sie duplikat klucza o roznych wartosciach, asercja na koncu
        funkcji zatrzyma skrypt zamiast pozwolic mu cicho wybrac losowy wiersz.
    """
    # `read_sql` wykonuje zapytanie SQL i pakuje wynik prosto w DataFrame.
    # Czytamy cala tabele - to kilkaset tysiecy wierszy, mieszcza sie w pamieci.
    tabela = pd.read_sql(f'SELECT * FROM "{nazwa}"', polaczenie)
    przed = len(tabela)

    # `drop_duplicates()` bez argumentow usuwa wiersze identyczne na WSZYSTKICH
    # kolumnach, zostawiajac pierwsze wystapienie.
    #
    # Dlaczego bez argumentow, a nie `drop_duplicates(subset=KLUCZ)`?
    # Wersja z `subset` usunelaby duplikaty klucza rowniez wtedy, gdyby wiersze
    # roznily sie wartosciami statystyk - i wybralaby jeden po cichu, bez
    # ostrzezenia. Sprawdzilem, ze w tej bazie duplikaty to dokladne kopie
    # (jeden mecz bca64d25, dwoch zawodnikow), wiec usuwanie po calym wierszu
    # daje ten sam wynik, ale jest bezpieczniejsze: gdyby dane sie zmienily i
    # duplikat okazal sie sprzeczny, oba wiersze przetrwaja i zlapie je asercja.
    tabela = tabela.drop_duplicates()
    usuniete = przed - len(tabela)

    # Deduplikacja to nie kosmetyka. Te dwa wiersze przechodza przez szesc
    # kolejnych polaczen tabel w ETAPIE 2. Bez usuniecia ich teraz kazde
    # laczenie mnozyloby je przez siebie (2 -> 4 -> 8 -> ...), co daje
    # kilkadziesiat falszywych wierszy dla dwoch zawodnikow.
    if usuniete:
        print(f"  {nazwa:20s} usunieto {usuniete} zdublowanych wierszy")

    # Sprawdzenie, a nie zalozenie: po deduplikacji klucz musi byc unikalny.
    #
    # Klucz zalezy od tabeli. Tabele wystepow opisuja "zawodnik w meczu", wiec
    # ich kluczem jest para (match_id, player_id). Tabela Match opisuje sam mecz
    # i nie ma kolumny player_id - jej kluczem jest samo match_id.
    klucz = [k for k in KLUCZ if k in tabela.columns]

    # `duplicated(subset=...)` zwraca serie True/False - True dla kazdego
    # wystapienia klucza poza pierwszym. Suma tej serii to liczba nadmiarowych
    # wierszy; ma byc zero.
    nadmiarowe = tabela.duplicated(subset=klucz).sum()
    assert nadmiarowe == 0, (
        f"{nazwa}: po deduplikacji nadal {nadmiarowe} zdublowanych kluczy "
        f"{klucz} - to znaczy, ze duplikaty NIE byly identycznymi kopiami "
        f"i trzeba recznie zdecydowac, ktory wiersz jest prawdziwy"
    )
    return tabela


def usun_redundancje(tabele: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Usuwa kolumny powielone miedzy tabelami i ujednoznacznia kolidujace nazwy.

    Przyjmuje:
        tabele - slownik {nazwa_tabeli: DataFrame} ze zdeduplikowanymi wierszami

    Zwraca:
        ten sam slownik, ale z wezszymi tabelami. Po tej operacji zadna nazwa
        kolumny (poza kluczem) nie powtarza sie miedzy tabelami polowymi, wiec
        polaczenie ich w ETAPIE 2 nie wyprodukuje sufiksow _x / _y.
    """
    for nazwa, do_usuniecia in KOLUMNY_ZREDUNDOWANE.items():
        # `drop(columns=...)` usuwa wskazane kolumny. Domyslnie rzuca bledem,
        # jesli ktorejs nie ma - i dobrze, bo to znak, ze schemat bazy sie
        # zmienil i lista redundancji jest nieaktualna. Nie uzywamy
        # `errors="ignore"`, ktore zamiotloby taka zmiane pod dywan.
        tabele[nazwa] = tabele[nazwa].drop(columns=do_usuniecia)
        print(f"  {nazwa:20s} usunieto {len(do_usuniecia)} zredundowanych kolumn")

    for nazwa, mapowanie in ZMIANY_NAZW.items():
        tabele[nazwa] = tabele[nazwa].rename(columns=mapowanie)
        for stara, nowa in mapowanie.items():
            print(f"  {nazwa:20s} zmiana nazwy: {stara} -> {nowa}")

    # Sprawdzenie: czy na pewno nie zostala zadna kolizja nazw?
    # Budujemy slownik {nazwa_kolumny: [tabele, w ktorych wystepuje]} i zadamy,
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
    ETAP 1: wczytuje z master.db wszystko, czego potrzebujemy, i doprowadza to
    do stanu, w ktorym mozna bezpiecznie laczyc tabele.

    Wczytujemy szesc tabel ze statystykami polowymi, tabele Player_Info (kto,
    ile minut, na jakiej pozycji) oraz Match (kiedy, w jakiej lidze, w jakim
    sezonie). Tabeli Goalkeeper nie ruszamy - bramkarzy odrzucamy z badania,
    bo ich wartosc rynkowa zalezy od zupelnie innych statystyk niz zawodnikow
    z pola, a jeden model dla obu grup mieszalby dwa rozne zjawiska.

    Przyjmuje:
        sciezka_db - sciezka do pliku master.db (otwierany tylko do odczytu)

    Zwraca:
        slownik {nazwa_tabeli: DataFrame}, gotowy dla ETAPU 2.
    """
    print("=" * 70)
    print("ETAP 1  wczytanie i deduplikacja tabel z master.db")
    print("=" * 70)

    assert sciezka_db.exists(), f"nie znaleziono bazy: {sciezka_db}"

    # Laczymy sie w trybie tylko do odczytu. `mode=ro` w adresie URI sprawia, ze
    # SQLite odmowi wykonania jakiegokolwiek zapisu - to techniczne
    # zabezpieczenie zasady "data/raw/ jest tylko do odczytu". Bez `uri=True`
    # sqlite3 potraktowalby ten ciag jako zwykla nazwe pliku i probowal utworzyc
    # nowa, pusta baze o absurdalnej nazwie.
    uri = f"file:{sciezka_db.as_posix()}?mode=ro"
    tabele: dict[str, pd.DataFrame] = {}

    # `closing` gwarantuje zamkniecie polaczenia nawet wtedy, gdy po drodze
    # poleci wyjatek. Samo `with sqlite3.connect(...)` tego NIE robi - ono
    # zarzadza transakcja, a nie polaczeniem, co jest czesta pomylka.
    from contextlib import closing

    with closing(sqlite3.connect(uri, uri=True)) as polaczenie:
        print("\n-- wczytywanie i deduplikacja wierszy --")
        for nazwa in [*TABELE_POLOWE, "Player_Info", "Match"]:
            tabele[nazwa] = wczytaj_tabele(polaczenie, nazwa)

        print("\n-- usuwanie kolumn powielonych miedzy tabelami --")
        tabele = usun_redundancje(tabele)

    # Kontrolka: chce widziec, z czym wchodze w ETAP 2.
    print("\n-- stan po ETAPIE 1 --")
    print(f"  {'tabela':22s} {'wierszy':>10s} {'kolumn':>8s}")
    for nazwa, tabela in tabele.items():
        print(f"  {nazwa:22s} {len(tabela):10d} {tabela.shape[1]:8d}")

    # Match ma wlasny klucz (match_id), nie zlozony - sprawdzamy go osobno.
    assert tabele["Match"]["match_id"].is_unique, "match_id nie jest unikalny w Match"

    # Wszystkie tabele polowe i Player_Info musza miec te sama liczbe wierszy,
    # bo opisuja dokladnie ten sam zbior wystepow zawodnikow w meczach.
    liczby = {n: len(tabele[n]) for n in [*TABELE_POLOWE, "Player_Info"]}
    assert len(set(liczby.values())) == 1, (
        f"tabele maja rozne liczby wierszy, wiec nie opisuja tego samego zbioru "
        f"wystepow: {liczby}"
    )
    print(f"\n  OK: wszystkie tabele wystepow maja po {len(tabele['Summary'])} wierszy")

    return tabele


# =====================================================================
# ETAP 2  polaczenie tabel po (match_id, player_id)
# =====================================================================


def etap_2_polacz_tabele(tabele: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    ETAP 2: skleja siedem tabel w jedna szeroka tabele wystepow.

    Po tym etapie jeden wiersz opisuje jeden wystep jednego zawodnika w jednym
    meczu, wraz z kompletem jego statystyk oraz informacja, kiedy i w jakich
    rozgrywkach ten mecz sie odbyl. To wciaz poziom meczu - sprowadzenie na
    poziom sezonu jest zadaniem ETAPU 4.

    Przyjmuje:
        tabele - slownik z ETAPU 1 (zdeduplikowany, bez kolizji nazw kolumn)

    Zwraca:
        DataFrame o unikalnym kluczu (match_id, player_id).

    Zaklada:
        ze wszystkie siedem tabel wystepow opisuje dokladnie ten sam zbior par
        (match_id, player_id). Sprawdzilem to zapytaniem EXCEPT w obie strony:
        zero roznic dla kazdej z szesciu tabel statystyk wzgledem Player_Info.
        Dlatego "inner" niczego tu nie gubi, a asercja po kazdym laczeniu
        pilnuje, zeby tak zostalo.
    """
    print("\n" + "=" * 70)
    print("ETAP 2  polaczenie tabel po (match_id, player_id)")
    print("=" * 70)

    # Zaczynamy od Player_Info, bo to ona mowi, KTO zagral - reszta tabel dodaje
    # tylko, JAK zagral. Gdybysmy zaczeli od Summary, mielibysmy statystyki
    # anonimowych identyfikatorow.
    polaczone = tabele["Player_Info"]
    oczekiwane = len(polaczone)
    print(f"\n  start: Player_Info, {oczekiwane} wystepow, {polaczone.shape[1]} kolumn")

    for nazwa in TABELE_POLOWE:
        # `merge` laczy dwie tabele po wspolnych kolumnach - odpowiednik JOIN
        # w SQL.
        #
        # on=KLUCZ mowi, ktore kolumny tworza klucz laczenia. Podajemy go jawnie
        # zamiast pozwolic pandas zgadywac po wspolnych nazwach - po ETAPIE 1
        # jedynymi wspolnymi nazwami sa wlasnie match_id i player_id, ale
        # poleganie na tym byloby uzaleznieniem sie od przypadku.
        #
        # validate="one_to_one" to najwazniejsze zabezpieczenie w calym skrypcie.
        # Pandas sprawdzi, czy kazdy klucz wystepuje DOKLADNIE RAZ po obu
        # stronach, i rzuci bledem, jesli nie. Bez tego argumentu duplikat klucza
        # po jednej stronie PO CICHU zwielokrotnia wiersze: dwa zdublowane
        # wystepy z ETAPU 1, przepuszczone przez szesc kolejnych laczen, uroslyby
        # do 2^7 = 128 falszywych wierszy dla dwoch zawodnikow. Nie byloby zadnego
        # komunikatu bledu - po prostu ci dwaj mieliby zawyzone sumy sezonowe.
        #
        # how="inner" zostawia tylko wiersze obecne po obu stronach. Alternatywy:
        #   "left"  zachowaloby wystepy bez statystyk, z pustymi kolumnami
        #   "outer" zachowaloby wszystko z obu stron
        # Tutaj chcemy inner, bo wystep bez kompletu statystyk jest dla modelu
        # bezuzyteczny. Skoro zbiory kluczy sa identyczne, inner i tak nie usunie
        # ani jednego wiersza - i wlasnie tego pilnuje asercja nizej.
        przed = polaczone.shape[1]
        polaczone = polaczone.merge(
            tabele[nazwa], on=KLUCZ, how="inner", validate="one_to_one"
        )

        # Sprawdzenie po kazdym laczeniu: liczba wierszy ma sie NIE zmienic.
        # Wzrost oznaczalby rozmnozenie przez duplikaty, spadek - ze ktoras
        # tabela nie zawiera wszystkich wystepow.
        assert len(polaczone) == oczekiwane, (
            f"po dolaczeniu {nazwa} jest {len(polaczone)} wierszy zamiast "
            f"{oczekiwane} - zbiory kluczy przestaly byc identyczne"
        )
        print(
            f"  + {nazwa:20s} {polaczone.shape[1] - przed:3d} nowych kolumn "
            f"-> {polaczone.shape[1]:3d} kolumn, {len(polaczone)} wierszy"
        )

    # ------------------------------------------------------------------
    # Kontekst meczu: kiedy i w jakich rozgrywkach
    # ------------------------------------------------------------------
    # Do tej pory wiemy, jak zawodnik zagral, ale nie wiemy w jakim sezonie ani
    # w jakiej lidze - a bez tego nie da sie ani zagregowac do sezonu (ETAP 4),
    # ani policzyc wieku na dzien meczu (ETAP 3), ani podzielic zbioru (ETAP 8).
    #
    # validate="many_to_one": po lewej stronie ten sam match_id powtarza sie dla
    # kazdego z ~28 zawodnikow bioracych udzial w meczu, ale po prawej musi
    # wystapic dokladnie raz. Gdyby tabela Match miala zdublowany mecz, kazdy
    # wystep w nim zostalby podwojony. Uzycie tu "one_to_one" byloby bledem -
    # ta walidacja wywalilaby sie na poprawnych danych, bo lewa strona z
    # zalozenia powtarza match_id dla kazdego zawodnika w meczu.
    przed = polaczone.shape[1]
    polaczone = polaczone.merge(
        tabele["Match"][KOLUMNY_Z_MATCH],
        on="match_id",
        how="inner",
        validate="many_to_one",
    )
    assert len(polaczone) == oczekiwane, (
        f"po dolaczeniu Match jest {len(polaczone)} wierszy zamiast {oczekiwane} "
        f"- w Match brakuje meczow, ktore wystepuja w Player_Info"
    )
    print(
        f"  + {'Match':20s} {polaczone.shape[1] - przed:3d} nowych kolumn "
        f"-> {polaczone.shape[1]:3d} kolumn, {len(polaczone)} wierszy"
    )

    # ------------------------------------------------------------------
    # Kontrolki koncowe
    # ------------------------------------------------------------------
    # Sufiksy _x / _y pojawiaja sie wtedy, gdy obie laczone tabele maja kolumne
    # o tej samej nazwie spoza klucza. Po ETAPIE 1 nie powinno ich byc wcale -
    # gdyby sie pojawily, znaczyloby to, ze lista redundancji jest niekompletna
    # i model dostalby te sama metryke dwa razy.
    sufiksy = [c for c in polaczone.columns if c.endswith(("_x", "_y"))]
    assert not sufiksy, f"laczenie wyprodukowalo zdublowane kolumny: {sufiksy}"

    # Klucz musi pozostac unikalny - to samo sprawdzenie co w ETAPIE 1, ale
    # teraz na scalonej tabeli.
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
