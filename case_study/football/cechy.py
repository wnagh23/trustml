"""
Przeksztalcenie danych FBref w cechy na poziomie sezonu.

ETAP 3  parsowanie: wiek, narodowosc, pozycja
ETAP 4  agregacja mecz -> zawodnik x sezon
ETAP 5  wskazniki procentowe i normalizacja per 90 minut
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from parametry import (
    KLUCZ_SEZON,
    KODY_AMERYKA_PLD,
    KODY_EUROPA,
    KOLUMNY_NIELICZNIKOWE,
    MAPA_POZYCJI,
    MIN_MINUT,
    SEZONY,
    WSKAZNIKI_PROCENTOWE,
)

# ETAP 3  parsowanie


def wybierz_najczestsza(
    ramka: pd.DataFrame, kolumna_klucza: str, kolumna_wartosci: str
) -> pd.Series:
    """
    Dla kazdego klucza wybiera wartosc najczestsza (dominanta).

    Uzywane do cech stalych zawodnika, gdzie odstepstwa sa bledami zrodla.
    Przy remisie wybiera mniejsza wartosc - wynik ma byc powtarzalny (W6).

    Przyjmuje:
        ramka            - dane z kolumna klucza i kolumna wartosci
        kolumna_klucza   - po czym grupujemy, np. "player_id"
        kolumna_wartosci - z czego wybieramy, np. "data_urodzenia"

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
    Odtwarza date urodzenia z wieku podanego na dzien meczu.

    FBref nie podaje daty urodzenia, ale podaje wiek w formacie lata-dni
    ("20-261") i date meczu, co jednoznacznie ja wyznacza. Liczymy z kazdego
    meczu osobno i bierzemy dominante, bo pojedyncze mecze miewaja blad.

    Przyjmuje:
        wystepy - tabela z kolumnami player_id, age, date

    Zwraca:
        Series: player_id -> data urodzenia. Zawodnicy bez ani jednego meczu
        z poprawnym wiekiem sie w niej nie znajda.
    """
    maska = wystepy["age"].notna()
    czastkowe = wystepy.loc[maska, ["player_id", "age", "date"]].copy()

    czesci = czastkowe["age"].str.split("-", expand=True)
    lata = czesci[0].astype(int)
    dni = czesci[1].astype(int)

    # Cofamy sie o dni do ostatnich urodzin, potem o pelne lata. Odejmowanie
    # 365 dni razy liczba lat rozjechaloby sie o lata przestepne.
    rocznica = czastkowe["date"] - pd.to_timedelta(dni, unit="D")
    czastkowe["data_urodzenia"] = pd.to_datetime(
        {
            "year": rocznica.dt.year - lata,
            "month": rocznica.dt.month,
            "day": rocznica.dt.day,
        },
        errors="coerce",
    )
    czastkowe = czastkowe.dropna(subset=["data_urodzenia"])

    # Ta sama data powinna wyjsc z kazdego meczu. Rozrzut oznacza blad zrodla.
    grupy = czastkowe.groupby("player_id")["data_urodzenia"]
    rozrzut = (grupy.max() - grupy.min()).dt.days
    niespojni = int((rozrzut > 0).sum())
    print(
        f"    odtworzona dla {len(rozrzut)} zawodnikow, {niespojni} ma niespojny "
        f"wiek miedzy meczami ({100 * niespojni / len(rozrzut):.2f}%)"
    )

    return wybierz_najczestsza(czastkowe, "player_id", "data_urodzenia")


def kod_na_region(kod) -> str | None:
    """
    Mapuje trzyliterowy kod kraju na region.

    Przyjmuje:
        kod - np. "BRA", "POL" albo brak

    Zwraca:
        "AMERYKA_PLD", "EUROPA", "RESZTA" albo None dla braku.
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
    Zamienia pola tekstowe FBref na cechy i odsiewa wiersze nieinterpretowalne.

    Wiek, narodowosc i pozycja przychodza jako tekst ("20-261", "eng ENG",
    "DM,CM"). Data urodzenia i narodowosc sa cechami stalymi zawodnika, wiec
    brak w jednym meczu uzupelniamy z pozostalych. Pozycja stala nie jest
    i propagacji nie podlega.

    Przyjmuje:
        wystepy - tabela z etapu 2

    Zwraca:
        te sama tabele, wezsza o odsiane wiersze i szersza o nowe kolumny.
    """
    print("\nETAP 3  parsowanie: wiek, narodowosc, pozycja")

    wystepy = wystepy.copy()
    na_wejsciu = len(wystepy)
    wystepy["date"] = pd.to_datetime(wystepy["date"])

    # Minuty sa mianownikiem kazdej statystyki per 90, wiec wystep o nieznanej
    # dlugosci skazilby wszystkie cechy zawodnika w sezonie. Czesc z nich ma
    # start=1, czyli sprzecznosc w zrodle - nie da sie zgadnac, ile zagral.
    bez_minut = wystepy["minutes"].isna()
    print(f"\n  odrzucam {bez_minut.sum()} wystepow bez zapisanych minut")
    wystepy = wystepy[~bez_minut]

    # --- wiek ---
    print("\n  wiek")
    # Date urodzenia liczymy przed odsianiem sezonow i bramkarzy - to cecha
    # metrykalna, wiec im wiecej meczow ja potwierdza, tym pewniejsza.
    daty_urodzenia = wylicz_daty_urodzenia(wystepy)
    wystepy["data_urodzenia"] = wystepy["player_id"].map(daty_urodzenia)

    bez_daty = wystepy["data_urodzenia"].isna()
    print(
        f"    bez daty mimo propagacji: {bez_daty.sum()} wystepow "
        f"({wystepy.loc[bez_daty, 'player_id'].nunique()} zawodnikow) - odrzucam"
    )
    wystepy = wystepy[~bez_daty]

    # Wiek na 30 czerwca roku konczacego sezon - ten sam moment, w ktorym
    # mierzymy wartosc rynkowa. Liczymy kalendarzowo, bez dzielenia przez 365,25:
    # urodzony w miesiacach I-VI mial juz urodziny przed 30 czerwca.
    rok_konca = wystepy["season"].str[-4:].astype(int)
    mial_juz_urodziny = wystepy["data_urodzenia"].dt.month <= 6
    wystepy["wiek"] = (
        rok_konca - wystepy["data_urodzenia"].dt.year - (~mial_juz_urodziny).astype(int)
    )
    print(
        f"    zakres {wystepy['wiek'].min()}-{wystepy['wiek'].max()} lat, "
        f"mediana {wystepy['wiek'].median():.0f}"
    )

    # --- narodowosc ---
    print("\n  narodowosc")
    # Pole nation wyglada jak "eng ENG" - bierzemy trzyliterowy kod po spacji.
    kody_z_meczow = wystepy["nation"].str.split().str[-1]
    pomocnicza = pd.DataFrame(
        {"player_id": wystepy["player_id"], "kod": kody_z_meczow}
    ).dropna()
    wystepy["kod_kraju"] = wystepy["player_id"].map(
        wybierz_najczestsza(pomocnicza, "player_id", "kod")
    )

    bez_kraju = wystepy["kod_kraju"].isna()
    print(
        f"    bez narodowosci mimo propagacji: {bez_kraju.sum()} wystepow "
        f"({wystepy.loc[bez_kraju, 'player_id'].nunique()} zawodnikow) - odrzucam"
    )
    wystepy = wystepy[~bez_kraju]

    wystepy["region"] = wystepy["kod_kraju"].map(kod_na_region)
    for nazwa_regionu, ile in wystepy["region"].value_counts().items():
        print(f"    {nazwa_regionu:12s} {ile:7d} ({100 * ile / len(wystepy):5.1f}%)")

    # Wypisujemy kody z kosza, zeby dalo sie zweryfikowac listy UEFA i CONMEBOL.
    reszta = sorted(wystepy.loc[wystepy["region"] == "RESZTA", "kod_kraju"].unique())
    print(f"    do RESZTY trafilo {len(reszta)} kodow: {' '.join(reszta)}")

    # --- pozycja ---
    print("\n  pozycja")
    # Pozycja bywa zlozona ("DM,CM") i kolejnosc ma znaczenie - FBref wymienia
    # najpierw pozycje podstawowa, wiec "FW,AM" to napastnik, a "AM,FW" pomocnik.
    pierwsza = wystepy["position"].str.split(",").str[0].str.strip()
    wystepy["pozycja_mecz"] = pierwsza.map(MAPA_POZYCJI)

    bramkarze = pierwsza == "GK"
    nieznane = wystepy["pozycja_mecz"].isna() & ~bramkarze
    print(f"    odrzucam {bramkarze.sum()} wystepow bramkarzy")
    print(
        f"    odrzucam {nieznane.sum()} o nieznanej pozycji: "
        f"{sorted(pierwsza[nieznane].unique())}"
    )
    wystepy = wystepy[wystepy["pozycja_mecz"].notna()]

    # --- sezony ---
    print("\n  sezony")
    do_analizy = wystepy["season"].isin(SEZONY)
    for sezon, ile in wystepy.loc[~do_analizy, "season"].value_counts().sort_index().items():
        print(f"    odrzucam {sezon}: {ile} wystepow")
    wystepy = wystepy[do_analizy]

    assert wystepy["season"].isin(SEZONY).all(), "zostal sezon spoza listy"
    assert wystepy["pozycja_mecz"].isin(["DEF", "MID", "FOR"]).all(), "obca pozycja"
    assert wystepy["minutes"].notna().all(), "zostal wystep bez minut"
    assert wystepy["wiek"].between(14, 50).all(), "wiek poza sensownym zakresem"

    odpadlo = na_wejsciu - len(wystepy)
    print(
        f"\n  stan po etapie: {len(wystepy)} wystepow "
        f"(odpadlo {odpadlo}, {100 * odpadlo / na_wejsciu:.1f}%), "
        f"{wystepy['player_id'].nunique()} zawodnikow"
    )
    return wystepy


# ETAP 4  agregacja do sezonu


def wybierz_wg_sumy_minut(wystepy: pd.DataFrame, kolumna: str) -> pd.Series:
    """
    Dla kazdej pary zawodnik-sezon wybiera wartosc o najwiekszej sumie minut.

    Minuty, nie liczba meczow: obronca, ktory 11 razy wszedl na 5 minut jako
    pomocnik i 10 razy zagral pelne 90 w obronie, wedlug meczow bylby pomocnikiem.
    Obie reguly roznia sie dla 2,9% par.

    Przyjmuje:
        wystepy - tabela na poziomie meczu, z kolumna "minutes"
        kolumna - co wybieramy, np. "pozycja_mecz" albo "competition"

    Zwraca:
        Series indeksowana (player_id, season).
    """
    minuty = wystepy.groupby(KLUCZ_SEZON + [kolumna])["minutes"].sum().reset_index()
    # Przy remisie decyduje kolejnosc alfabetyczna - zeby wynik byl powtarzalny.
    minuty = minuty.sort_values(
        KLUCZ_SEZON + ["minutes", kolumna], ascending=[True, True, False, True]
    )
    najlepsze = minuty.drop_duplicates(subset=KLUCZ_SEZON, keep="first")
    return najlepsze.set_index(KLUCZ_SEZON)[kolumna]


def etap_4_agreguj(wystepy: pd.DataFrame) -> pd.DataFrame:
    """
    Sprowadza dane z poziomu meczu na poziom sezonu.

    Model ocenia zawodnika za sezon, bo wartosc rynkowa odzwierciedla caloroczny
    dorobek. Kolumny procentowe sa tu odrzucane - policzy je etap 5 z liczników
    i mianownikow, zeby nie bylo pokusy usredniania.

    Przyjmuje:
        wystepy - tabela z etapu 3

    Zwraca:
        DataFrame o kluczu (player_id, season) z surowymi sumami sezonowymi.
    """
    print("\nETAP 4  agregacja mecz -> zawodnik x sezon")

    na_wejsciu = len(wystepy)

    # Statystyki do zsumowania to "cala reszta" - wszystko poza metadanymi
    # i kolumnami procentowymi.
    licznikowe = [
        c
        for c in wystepy.columns
        if c not in KOLUMNY_NIELICZNIKOWE and c not in WSKAZNIKI_PROCENTOWE
    ]
    for procent, (licznik, mianowniki) in WSKAZNIKI_PROCENTOWE.items():
        for potrzebna in [licznik, *mianowniki]:
            assert potrzebna in licznikowe, (
                f"{potrzebna}, potrzebna do {procent}, nie trafila do sumowania"
            )

    print(f"\n  sumuje {len(licznikowe)} statystyk, "
          f"odrzucam {len(WSKAZNIKI_PROCENTOWE)} kolumn procentowych")

    sezony = wystepy.groupby(KLUCZ_SEZON, as_index=False)[licznikowe].sum()

    podsumowanie = wystepy.groupby(KLUCZ_SEZON, as_index=False).agg(
        minuty_sezon=("minutes", "sum"),
        mecze=("match_id", "size"),
        mecze_od_poczatku=("start", "sum"),
    )

    # Cechy stale w obrebie pary - bierzemy pierwsza wartosc z grupy.
    stale = wystepy.groupby(KLUCZ_SEZON, as_index=False).agg(
        name=("name", "first"),
        data_urodzenia=("data_urodzenia", "first"),
        wiek=("wiek", "first"),
        kod_kraju=("kod_kraju", "first"),
        region=("region", "first"),
    )
    for kolumna in ["data_urodzenia", "wiek", "kod_kraju", "region"]:
        ile_roznych = wystepy.groupby(KLUCZ_SEZON)[kolumna].nunique()
        assert (ile_roznych <= 1).all(), f"{kolumna} nie jest stala w obrebie pary"

    pozycja = wybierz_wg_sumy_minut(wystepy, "pozycja_mecz").rename("pozycja")
    liga = wybierz_wg_sumy_minut(wystepy, "competition").rename("liga")

    # Slad po transferze w trakcie sezonu - przyda sie przy analizie W5.
    ile_lig = wystepy.groupby(KLUCZ_SEZON)["competition"].nunique()
    zmienil = (ile_lig > 1).rename("zmienil_lige")
    print(f"  par grajacych w wiecej niz jednej lidze: {zmienil.sum()} "
          f"({100 * zmienil.mean():.1f}%)")

    sezony = sezony.merge(podsumowanie, on=KLUCZ_SEZON, validate="one_to_one")
    sezony = sezony.merge(stale, on=KLUCZ_SEZON, validate="one_to_one")
    for dodatkowa in [pozycja, liga, zmienil]:
        sezony = sezony.merge(
            dodatkowa.reset_index(), on=KLUCZ_SEZON, validate="one_to_one"
        )

    dosc_minut = sezony["minuty_sezon"] >= MIN_MINUT
    print(f"  odrzucam {(~dosc_minut).sum()} par ponizej {MIN_MINUT} minut "
          f"({100 * (~dosc_minut).mean():.1f}%)")
    sezony = sezony[dosc_minut]

    assert not sezony.duplicated(subset=KLUCZ_SEZON).any(), "klucz nie jest unikalny"
    assert (sezony["minuty_sezon"] >= MIN_MINUT).all(), "para ponizej progu"
    assert sezony["pozycja"].isin(["DEF", "MID", "FOR"]).all(), "obca pozycja"
    assert (sezony["mecze_od_poczatku"] <= sezony["mecze"]).all(), (
        "wiecej wyjsc w pierwszym skladzie niz meczow"
    )
    assert (sezony["minuty_sezon"] <= 90 * sezony["mecze"]).all(), (
        "suma minut przekracza 90 na mecz"
    )

    print(f"\n  stan po etapie: {na_wejsciu} wystepow -> {len(sezony)} par, "
          f"{sezony['player_id'].nunique()} zawodnikow")
    rozklad = sezony["pozycja"].value_counts()
    print("    " + "  ".join(
        f"{poz} {ile} ({100 * ile / len(sezony):.0f}%)" for poz, ile in rozklad.items()
    ))
    return sezony


# ETAP 5  wskazniki i normalizacja


def policz_wskazniki_procentowe(sezony: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Liczy wskazniki jako suma_licznikow / suma_mianownikow i tnie do [0, 100].

    Zerowy mianownik daje NaN i tak ma zostac - to informacja "nie probowal ani
    razu", nie brak do imputacji. Wpisanie zera znaczyloby "probowal i zawsze
    przegrywal", a mediany - "byl przecietny".

    Przyjmuje:
        sezony - tabela z zsumowanymi licznikami i mianownikami

    Zwraca:
        (tabela z kolumnami procentowymi, slownik przycietych wierszy)
    """
    przyciete = {}

    for nazwa, (licznik, mianowniki) in WSKAZNIKI_PROCENTOWE.items():
        # sum(axis=1) dodaje kolumny w obrebie wiersza - mianownik pojedynkow
        # powietrznych powstaje z dwoch kolumn.
        mianownik = sezony[mianowniki].sum(axis=1)
        # Zerowy mianownik zamieniamy na NaN przed dzieleniem, zeby nie dostac
        # nieskonczonosci ani ostrzezen numpy.
        sezony[nazwa] = 100 * sezony[licznik] / mianownik.where(mianownik > 0)

        poza = ((sezony[nazwa] < 0) | (sezony[nazwa] > 100)).sum()
        if poza:
            przyciete[nazwa] = int(poza)
        sezony[nazwa] = sezony[nazwa].clip(lower=0, upper=100)

    return sezony, przyciete


def etap_5_normalizuj(sezony: pd.DataFrame) -> pd.DataFrame:
    """
    Dolicza wskazniki procentowe i przelicza statystyki na 90 minut.

    Sama suma sezonowa miesza jakosc zawodnika z liczba rozegranych minut:
    10 goli w 3400 minut to gorszy strzelec niz 8 goli w 900 minut, a suma mowi
    odwrotnie. Surowe sumy zostaja w tabeli jako material kontrolny.

    Przyjmuje:
        sezony - tabela z etapu 4

    Zwraca:
        te sama tabele, poszerzona o kolumny procentowe i _p90.
    """
    print("\nETAP 5  wskazniki procentowe i normalizacja per 90 minut")

    sezony = sezony.copy()
    sezony, przyciete = policz_wskazniki_procentowe(sezony)

    print()
    for nazwa in WSKAZNIKI_PROCENTOWE:
        braki = sezony[nazwa].isna().sum()
        print(f"  {nazwa:40s} mediana {sezony[nazwa].median():5.1f}%   "
              f"NaN {braki:4d} ({100 * braki / len(sezony):4.1f}%)")

    if przyciete:
        for nazwa, ile in przyciete.items():
            print(f"  przyciete do [0,100]: {nazwa} - {ile} wierszy")
    else:
        print("\n  przycinanie do [0,100]: nic nie wyszlo poza zakres")

    # Ile "pelnych meczow" rozegral zawodnik. Prog minutowy z etapu 4 gwarantuje,
    # ze mianownik jest dodatni.
    mecze_po_90 = sezony["minuty_sezon"] / 90
    licznikowe = [
        c
        for c in sezony.columns
        if c not in KOLUMNY_NIELICZNIKOWE and c not in WSKAZNIKI_PROCENTOWE
    ]
    # Budujemy wszystkie kolumny naraz - dokladanie ich po jednej kopiowaloby
    # cala tabele przy kazdym przypisaniu.
    p90 = pd.DataFrame(
        {f"{kolumna}_p90": sezony[kolumna] / mecze_po_90 for kolumna in licznikowe},
        index=sezony.index,
    )
    sezony = pd.concat([sezony, p90], axis=1)

    for nazwa in WSKAZNIKI_PROCENTOWE:
        assert sezony[nazwa].dropna().between(0, 100).all(), f"{nazwa} poza [0,100]"
    kolumny_p90 = [c for c in sezony.columns if c.endswith("_p90")]
    assert np.isfinite(sezony[kolumny_p90].to_numpy()).all(), (
        "nieskonczonosc w kolumnach _p90 - dzielenie przez zero"
    )
    assert sezony["goals_p90"].max() < 5, "nierealny wynik goli na 90 minut"

    print(f"\n  przeliczono {len(licznikowe)} statystyk na 90 minut")
    print(f"  kontrola: gole/90 mediana {sezony['goals_p90'].median():.2f}, "
          f"maks {sezony['goals_p90'].max():.2f}")
    print(f"  stan po etapie: {len(sezony)} par, {sezony.shape[1]} kolumn")
    return sezony
