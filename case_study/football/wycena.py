"""
Polaczenie z Transfermarktem i dolaczenie zmiennej celu.

ETAP 6  crosswalk FBref -> TM, cechy statyczne
ETAP 7  wycena z okna IV-IX, najblizsza 30.06
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from parametry import (
    DZIEN_ODNIESIENIA,
    KLUCZ_SEZON,
    MAX_ROZNICA_DAT_DNI,
    OKNO_WYCENY_DO,
    OKNO_WYCENY_OD,
    SCIEZKA_TM,
)

# ETAP 6  crosswalk


def znormalizuj_nazwe(seria: pd.Series) -> pd.Series:
    """
    Sprowadza nazwiska do postaci porownywalnej miedzy zrodlami.

    Bez znakow diakrytycznych, male litery, tylko litery i spacje.

    Przyjmuje:
        seria - kolumna z nazwiskami

    Zwraca:
        kolumne znormalizowanych nazw.
    """
    # NFKD rozklada znak z akcentem na litere i akcent, a encode go wyrzuca.
    rozlozone = seria.astype(str).map(lambda t: unicodedata.normalize("NFKD", t))
    bez_akcentow = rozlozone.str.encode("ascii", "ignore").str.decode("ascii")
    oczyszczone = bez_akcentow.str.lower().str.replace(r"[^a-z ]", "", regex=True)
    return oczyszczone.str.replace(r"\s+", " ", regex=True).str.strip()


def dopasuj_po_kluczu(
    fbref_wolni: pd.DataFrame,
    tm_wolne: pd.DataFrame,
    klucz_fb: list[str],
    klucz_tm: list[str],
) -> pd.DataFrame:
    """
    Jeden krok kaskady: laczy po podanym kluczu, pomijajac kolizje.

    Wartosc klucza wystepujaca po ktorejkolwiek stronie wiecej niz raz odrzucamy
    w calosci - lepiej zostawic zawodnika nastepnemu krokowi niz przypisac mu
    cudza wycene.

    Przyjmuje:
        fbref_wolni - zawodnicy jeszcze niedopasowani
        tm_wolne    - rekordy TM jeszcze nieprzypisane, z kolumna tm_player_id
        klucz_fb    - kolumny klucza po stronie FBref
        klucz_tm    - odpowiadajace im kolumny po stronie TM

    Zwraca:
        DataFrame z kolumnami player_id i tm_player_id.
    """
    fb = fbref_wolni.dropna(subset=klucz_fb).drop_duplicates(subset=klucz_fb, keep=False)
    tm = tm_wolne.dropna(subset=klucz_tm).drop_duplicates(subset=klucz_tm, keep=False)

    return fb.merge(
        tm[[*klucz_tm, "tm_player_id"]],
        left_on=klucz_fb,
        right_on=klucz_tm,
        how="inner",
        validate="one_to_one",
    )[["player_id", "tm_player_id"]]


def zbuduj_crosswalk(sezony: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    """
    Mapowanie zawodnik FBref -> zawodnik TM, kaskada trzystopniowa.

    Kazdy klucz zawodzi gdzie indziej, wiec sie uzupelniaja:
        1. nazwa + data urodzenia - najpewniejszy, rozroznia imiennikow
        2. data urodzenia + kraj  - nie oglada sie na nazwe, lapie pseudonimy
        3. sama nazwa             - reszta

    Najpewniejszy klucz idzie pierwszy, zeby przy konflikcie to on decydowal.
    Pokrycie 97,5% wobec 83,7% dla samej nazwy.

    Przyjmuje:
        sezony - tabela z etapu 5
        tm     - wczytany players.csv

    Zwraca:
        DataFrame z kolumnami player_id i tm_player_id.
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

    print(f"\n  zawodnikow FBref: {len(fbref)}, rekordow TM: {len(tm)}")

    czesci: list[pd.DataFrame] = []
    uzyci_fb: set = set()
    uzyte_tm: set = set()

    def wolni():
        return (
            fbref[~fbref["player_id"].isin(uzyci_fb)],
            tm[~tm["tm_player_id"].isin(uzyte_tm)],
        )

    def zapisz(wynik: pd.DataFrame, opis: str):
        czesci.append(wynik)
        uzyci_fb.update(wynik["player_id"])
        uzyte_tm.update(wynik["tm_player_id"])
        print(f"    {opis:28s} {len(wynik):5d} zawodnikow")

    fb_wolni, tm_wolne = wolni()
    zapisz(
        dopasuj_po_kluczu(
            fb_wolni, tm_wolne,
            ["nazwa_norm", "data_urodzenia"],
            ["nazwa_norm", "data_urodzenia_tm"],
        ),
        "1. nazwa + data urodzenia",
    )

    # Kraje zapisane sa inaczej po obu stronach: "BRA" kontra "Brazil". Mape
    # wyprowadzamy z dopasowan kroku 1 zamiast wpisywac 143 kraje recznie.
    # Do nauczenia kodu wystarczy dowolny zawodnik z danego kraju, wiec Henrique
    # nie musi tam byc - potem tylko z mapy korzysta.
    krok1 = czesci[0]
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

    bez_mapy = sorted(set(fbref.loc[fbref["kraj_tm"].isna(), "kod_kraju"].dropna()))
    if bez_mapy:
        print(f"    kodow bez odpowiednika w mapie: {len(bez_mapy)} {bez_mapy}")

    fb_wolni, tm_wolne = wolni()
    zapisz(
        dopasuj_po_kluczu(
            fb_wolni, tm_wolne,
            ["data_urodzenia", "kraj_tm"],
            ["data_urodzenia_tm", "country_of_citizenship"],
        ),
        "2. data urodzenia + kraj",
    )

    fb_wolni, tm_wolne = wolni()
    zapisz(
        dopasuj_po_kluczu(fb_wolni, tm_wolne, ["nazwa_norm"], ["nazwa_norm"]),
        "3. sama nazwa",
    )

    crosswalk = pd.concat(czesci, ignore_index=True)
    assert not crosswalk["player_id"].duplicated().any(), "zawodnik dopasowany dwa razy"
    assert not crosswalk["tm_player_id"].duplicated().any(), (
        "ten sam rekord TM przypisany dwom zawodnikom"
    )

    print(f"    razem: {len(crosswalk)} z {len(fbref)} "
          f"({100 * len(crosswalk) / len(fbref):.1f}%)")
    return crosswalk


def etap_6_crosswalk(sezony: pd.DataFrame) -> pd.DataFrame:
    """
    Laczy zawodnikow z Transfermarktem i dokleja cechy statyczne.

    Statystyki mamy z FBref, ale zmiennej celu tam nie ma. Oba zrodla uzywaja
    wlasnych identyfikatorow, ktorych nic nie laczy poza nazwiskiem i data
    urodzenia.

    Przyjmuje:
        sezony - tabela z etapu 5

    Zwraca:
        tabele bez par bez dopasowania, z tm_player_id, wzrostem i noga.
    """
    print("\nETAP 6  crosswalk do Transfermarktu i cechy statyczne")

    na_wejsciu = len(sezony)
    przed_crosswalkiem = sezony
    tm = pd.read_csv(SCIEZKA_TM / "players.csv")

    crosswalk = zbuduj_crosswalk(sezony, tm)
    sezony = sezony.merge(crosswalk, on="player_id", how="inner", validate="many_to_one")
    print(f"\n  par: {na_wejsciu} -> {len(sezony)} "
          f"({100 * len(sezony) / na_wejsciu:.1f}% pokrycia)")

    # Utrata nie jest calkiem losowa, choc kaskada mocno ja zmniejszyla.
    # Zawodnicy brazylijscy i portugalscy wystepuja pod jednoczlonowymi
    # pseudonimami, ktore czesciej koliduja. Ma to znaczenie dla W2 (zbior_C)
    # i W5 (region jako atrybut chroniony), wiec raportujemy przy kazdym
    # uruchomieniu.
    print("  utrata wg ligi i regionu:")
    for kolumna in ["liga", "region"]:
        przed = przed_crosswalkiem[kolumna].value_counts()
        po = sezony[kolumna].value_counts().reindex(przed.index).fillna(0)
        udzial = (100 * (1 - po / przed)).sort_values(ascending=False)
        opis = "  ".join(f"{k} {v:.1f}%" for k, v in udzial.items())
        print(f"    {opis}")

    # Z TM bierzemy tylko to, czego FBref nie ma. Pozycji i narodowosci nie -
    # mamy wlasne, liczone z rzeczywistych minut.
    cechy = tm[["player_id", "height_in_cm", "foot", "date_of_birth"]].rename(
        columns={
            "player_id": "tm_player_id",
            "height_in_cm": "wzrost_cm",
            "foot": "noga",
            "date_of_birth": "data_urodzenia_tm",
        }
    )
    cechy["data_urodzenia_tm"] = pd.to_datetime(cechy["data_urodzenia_tm"], errors="coerce")
    sezony = sezony.merge(cechy, on="tm_player_id", how="left", validate="many_to_one")

    print(f"\n  wzrost_cm: brak u {sezony['wzrost_cm'].isna().sum()} par, "
          f"mediana {sezony['wzrost_cm'].median():.0f} cm")
    print(f"  noga: brak u {sezony['noga'].isna().sum()} par, "
          f"{dict(sezony['noga'].value_counts())}")

    # Data urodzenia jest niezaleznym swiadkiem poprawnosci dopasowania: FBref
    # liczy ja z wieku przy meczu, TM podaje wprost. Duza rozbieznosc oznacza
    # innego zawodnika, czyli cudza wycene w wierszu.
    roznica = (sezony["data_urodzenia"] - sezony["data_urodzenia_tm"]).dt.days.abs()
    niezgodne = (roznica > 0).sum()
    print(f"\n  zgodnosc dat urodzenia: {niezgodne} niezgodnych "
          f"({100 * niezgodne / roznica.notna().sum():.2f}%)")

    # Porownanie z NaN daje False, wiec pary bez daty w TM zostaja - nie mamy
    # podstaw, zeby je odrzucic.
    bledne = (roznica > MAX_ROZNICA_DAT_DNI).reindex(sezony.index, fill_value=False)
    print(f"  odrzucam {bledne.sum()} par o rozbieznosci > {MAX_ROZNICA_DAT_DNI} dni "
          f"({sezony.loc[bledne, 'player_id'].nunique()} zawodnikow)")
    sezony = sezony[~bledne]

    assert not sezony.duplicated(subset=KLUCZ_SEZON).any(), "klucz nie jest unikalny"
    przypisania = sezony.drop_duplicates(subset="player_id")
    assert not przypisania["tm_player_id"].duplicated().any(), (
        "ten sam zawodnik TM przypisany dwom zawodnikom FBref"
    )
    # W players.csv zdarzaja sie wzrosty rzedu 17 cm. Do naszego zbioru zaden
    # taki nie trafil i ta asercja tego pilnuje.
    obecne = sezony["wzrost_cm"].dropna()
    assert obecne.between(150, 215).all(), f"nierealny wzrost: {obecne.min()}-{obecne.max()}"

    print(f"\n  stan po etapie: {len(sezony)} par, "
          f"{sezony['player_id'].nunique()} zawodnikow")
    return sezony


# ETAP 7  wycena


def etap_7_dolacz_wycene(sezony: pd.DataFrame) -> pd.DataFrame:
    """
    Dokleja zmienna objasniana - wartosc rynkowa z Transfermarktu.

    To model wyceny biezacej, nie prognoza: cechy opisuja sezon konczacy sie
    w roku t, a cel pochodzi z okolic 30 czerwca tego samego roku.

    Przyjmuje:
        sezony - tabela z etapu 6

    Zwraca:
        tabele bez par bez wyceny, z kolumna market_value_in_eur.
    """
    print("\nETAP 7  dolaczenie wyceny (okno IV-IX, najblizsza 30.06)")

    na_wejsciu = len(sezony)
    sezony = sezony.copy()

    wyceny = pd.read_csv(SCIEZKA_TM / "player_valuations.csv")
    wyceny["date"] = pd.to_datetime(wyceny["date"])
    wyceny = wyceny.rename(columns={"player_id": "tm_player_id"})
    # Zawezenie przed laczeniem - inaczej 508 tys. wycen probowaloby sie polaczyc
    # z kazda para.
    wyceny = wyceny[wyceny["tm_player_id"].isin(sezony["tm_player_id"])]

    sezony["rok_t"] = sezony["season"].str[-4:].astype(int)

    # Laczymy tylko klucz, nie cala szeroka tabele - inaczej kazda z 200 kolumn
    # zostalaby powielona tyle razy, ile zawodnik ma wycen. To laczenie jest
    # z zalozenia wiele-do-wielu, wiec bez validate.
    kandydaci = sezony[["tm_player_id", "season", "rok_t"]].merge(
        wyceny[["tm_player_id", "date", "market_value_in_eur"]],
        on="tm_player_id",
        how="inner",
    )

    rok_tekst = kandydaci["rok_t"].astype(str)
    poczatek = pd.to_datetime(rok_tekst + OKNO_WYCENY_OD)
    koniec = pd.to_datetime(rok_tekst + OKNO_WYCENY_DO)
    odniesienie = pd.to_datetime(rok_tekst + DZIEN_ODNIESIENIA)

    w_oknie = kandydaci[
        (kandydaci["date"] >= poczatek) & (kandydaci["date"] <= koniec)
    ].copy()
    w_oknie["dni_od_30_06"] = (w_oknie["date"] - odniesienie[w_oknie.index]).dt.days
    w_oknie["odleglosc"] = w_oknie["dni_od_30_06"].abs()

    # Przy jednakowej odleglosci decyduje wczesniejsza data. Zmierzone: remisow
    # nie ma, ale bez tego warunku wynik zalezalby od kolejnosci wierszy w CSV.
    w_oknie = w_oknie.sort_values(
        ["tm_player_id", "season", "odleglosc", "date"]
    )
    wybrane = w_oknie.drop_duplicates(subset=["tm_player_id", "season"], keep="first")

    print(f"\n  par z wycena w oknie: {len(wybrane)} z {na_wejsciu} "
          f"({100 * len(wybrane) / na_wejsciu:.1f}%)")
    print(f"  odleglosc od 30.06: mediana {wybrane['odleglosc'].median():.0f} dni, "
          f"maks {wybrane['odleglosc'].max()}")

    sezony = sezony.merge(
        wybrane[["tm_player_id", "season", "market_value_in_eur", "date",
                 "dni_od_30_06"]].rename(columns={"date": "data_wyceny"}),
        on=["tm_player_id", "season"],
        how="inner",
        validate="one_to_one",
    )

    # Wycena zerowa nie ma sensu ekonomicznego, a po zlogarytmowaniu bylaby
    # nieodrozniania od zawodnika wartego 1 euro.
    niedodatnie = (sezony["market_value_in_eur"] <= 0).sum()
    if niedodatnie:
        print(f"  odrzucam {niedodatnie} par z wycena <= 0")
        sezony = sezony[sezony["market_value_in_eur"] > 0]

    assert (sezony["data_wyceny"].dt.year == sezony["rok_t"]).all(), (
        "wycena z innego roku niz rok konczacy sezon"
    )
    assert sezony["data_wyceny"].dt.month.between(4, 9).all(), "wycena spoza okna IV-IX"
    assert not sezony.duplicated(subset=KLUCZ_SEZON).any(), "klucz nie jest unikalny"

    print(f"\n  stan po etapie: {len(sezony)} par")
    print(f"    mediana wyceny {sezony['market_value_in_eur'].median():>12,.0f} EUR")
    print(f"    srednia        {sezony['market_value_in_eur'].mean():>12,.0f} EUR")
    # Surowa mediana nie nadaje sie do sledzenia inflacji - TM wycenia
    # w okraglych progach, wiec stoi na tym samym progu przez kilka sezonow.
    print("\n    sezon        mediana      srednia   sr. log")
    for sezon, grupa in sezony.groupby("season"):
        print(f"    {sezon}  {grupa['market_value_in_eur'].median():>11,.0f} "
              f"{grupa['market_value_in_eur'].mean():>12,.0f} "
              f"{np.log(grupa['market_value_in_eur']).mean():>9.3f}")

    return sezony
