"""
Etap 3: parsowanie wieku, narodowosci i pozycji.
Etap 4: agregacja mecz -> zawodnik x sezon.
Etap 5: wskazniki procentowe i normalizacja per 90 minut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .parametry import (
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
    """Najczestsza wartosc dla kazdego klucza; przy remisie mniejsza wartosc."""
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
    Odtwarza date urodzenia z wieku "lata-dni" i daty meczu.

    Licze z kazdego meczu i biore najczestsza date. Zwraca player_id -> data.
    """
    maska = wystepy["age"].notna()
    czastkowe = wystepy.loc[maska, ["player_id", "age", "date"]].copy()

    czesci = czastkowe["age"].str.split("-", expand=True)
    lata = czesci[0].astype(int)
    dni = czesci[1].astype(int)

    # najpierw cofam o dni do ostatnich urodzin, potem o pelne lata (lata przestepne)
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

    grupy = czastkowe.groupby("player_id")["data_urodzenia"]
    rozrzut = (grupy.max() - grupy.min()).dt.days
    niespojni = int((rozrzut > 0).sum())
    print(
        f"    odtworzona dla {len(rozrzut)} zawodnikow, {niespojni} ma niespojny "
        f"wiek miedzy meczami ({100 * niespojni / len(rozrzut):.2f}%)"
    )

    return wybierz_najczestsza(czastkowe, "player_id", "data_urodzenia")


def kod_na_region(kod) -> str | None:
    """Kod kraju -> "AMERYKA_PLD", "EUROPA", "RESZTA" albo None."""
    if pd.isna(kod):
        return None
    if kod in KODY_AMERYKA_PLD:
        return "AMERYKA_PLD"
    if kod in KODY_EUROPA:
        return "EUROPA"
    return "RESZTA"


def etap_3_parsuj(wystepy: pd.DataFrame) -> pd.DataFrame:
    """Parsuje wiek, narodowosc i pozycje; odrzuca wiersze, ktorych nie da sie uzyc."""
    print("\nETAP 3  parsowanie: wiek, narodowosc, pozycja")

    wystepy = wystepy.copy()
    na_wejsciu = len(wystepy)
    wystepy["date"] = pd.to_datetime(wystepy["date"])

    # minuty sa mianownikiem kazdej cechy per 90
    bez_minut = wystepy["minutes"].isna()
    print(f"\n  odrzucam {bez_minut.sum()} wystepow bez zapisanych minut")
    wystepy = wystepy[~bez_minut]

    # --- wiek ---
    print("\n  wiek")
    # date urodzenia licze przed filtrami sezonow, wiecej meczow = pewniejszy wynik
    daty_urodzenia = wylicz_daty_urodzenia(wystepy)
    wystepy["data_urodzenia"] = wystepy["player_id"].map(daty_urodzenia)

    bez_daty = wystepy["data_urodzenia"].isna()
    print(
        f"    bez daty mimo propagacji: {bez_daty.sum()} wystepow "
        f"({wystepy.loc[bez_daty, 'player_id'].nunique()} zawodnikow) - odrzucam"
    )
    wystepy = wystepy[~bez_daty]

    # wiek na 30 czerwca roku konczacego sezon, czyli w momencie wyceny
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
    # pole nation wyglada jak "eng ENG"
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

    reszta = sorted(wystepy.loc[wystepy["region"] == "RESZTA", "kod_kraju"].unique())
    print(f"    do RESZTY trafilo {len(reszta)} kodow: {' '.join(reszta)}")

    # --- pozycja ---
    print("\n  pozycja")
    # pierwsza pozycja jest podstawowa: "FW,AM" to napastnik, "AM,FW" pomocnik
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
    for sezon, ile in (
        wystepy.loc[~do_analizy, "season"].value_counts().sort_index().items()
    ):
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
    Dla kazdej pary zawodnik-sezon wybiera wartosc z najwieksza suma minut.

    Uzywane do pozycji i ligi. Zwraca Series z indeksem (player_id, season).
    """
    minuty = wystepy.groupby(KLUCZ_SEZON + [kolumna])["minutes"].sum().reset_index()
    # przy remisie decyduje kolejnosc alfabetyczna
    minuty = minuty.sort_values(
        KLUCZ_SEZON + ["minutes", kolumna], ascending=[True, True, False, True]
    )
    najlepsze = minuty.drop_duplicates(subset=KLUCZ_SEZON, keep="first")
    return najlepsze.set_index(KLUCZ_SEZON)[kolumna]


def etap_4_agreguj(wystepy: pd.DataFrame) -> pd.DataFrame:
    """
    Agreguje wystepy do par (player_id, season) i odrzuca pary ponizej MIN_MINUT.

    Kolumny procentowe pomijam, etap 5 liczy je z zsumowanych licznikow.
    """
    print("\nETAP 4  agregacja mecz -> zawodnik x sezon")

    na_wejsciu = len(wystepy)

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

    print(
        f"\n  sumuje {len(licznikowe)} statystyk, "
        f"odrzucam {len(WSKAZNIKI_PROCENTOWE)} kolumn procentowych"
    )

    sezony = wystepy.groupby(KLUCZ_SEZON, as_index=False)[licznikowe].sum()

    podsumowanie = wystepy.groupby(KLUCZ_SEZON, as_index=False).agg(
        minuty_sezon=("minutes", "sum"),
        mecze=("match_id", "size"),
        mecze_od_poczatku=("start", "sum"),
    )

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

    ile_lig = wystepy.groupby(KLUCZ_SEZON)["competition"].nunique()
    zmienil = (ile_lig > 1).rename("zmienil_lige")
    print(
        f"  par grajacych w wiecej niz jednej lidze: {zmienil.sum()} "
        f"({100 * zmienil.mean():.1f}%)"
    )

    sezony = sezony.merge(podsumowanie, on=KLUCZ_SEZON, validate="one_to_one")
    sezony = sezony.merge(stale, on=KLUCZ_SEZON, validate="one_to_one")
    for dodatkowa in [pozycja, liga, zmienil]:
        sezony = sezony.merge(
            dodatkowa.reset_index(), on=KLUCZ_SEZON, validate="one_to_one"
        )

    dosc_minut = sezony["minuty_sezon"] >= MIN_MINUT
    print(
        f"  odrzucam {(~dosc_minut).sum()} par ponizej {MIN_MINUT} minut "
        f"({100 * (~dosc_minut).mean():.1f}%)"
    )
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

    print(
        f"\n  stan po etapie: {na_wejsciu} wystepow -> {len(sezony)} par, "
        f"{sezony['player_id'].nunique()} zawodnikow"
    )
    rozklad = sezony["pozycja"].value_counts()
    print(
        "    "
        + "  ".join(
            f"{poz} {ile} ({100 * ile / len(sezony):.0f}%)"
            for poz, ile in rozklad.items()
        )
    )
    return sezony


# ETAP 5  wskazniki i normalizacja


def policz_wskazniki_procentowe(sezony: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Liczy wskazniki jako suma licznikow / suma mianownikow i przycina do [0, 100].

    Zerowy mianownik daje NaN (zero prob). Zwraca (tabela, liczba przycietych).
    """
    przyciete = {}

    for nazwa, (licznik, mianowniki) in WSKAZNIKI_PROCENTOWE.items():
        mianownik = sezony[mianowniki].sum(axis=1)
        sezony[nazwa] = 100 * sezony[licznik] / mianownik.where(mianownik > 0)

        poza = ((sezony[nazwa] < 0) | (sezony[nazwa] > 100)).sum()
        if poza:
            przyciete[nazwa] = int(poza)
        sezony[nazwa] = sezony[nazwa].clip(lower=0, upper=100)

    return sezony, przyciete


def etap_5_normalizuj(sezony: pd.DataFrame) -> pd.DataFrame:
    """Dodaje wskazniki procentowe i kolumny _p90 (suma sezonowa / (minuty / 90))."""
    print("\nETAP 5  wskazniki procentowe i normalizacja per 90 minut")

    sezony = sezony.copy()
    sezony, przyciete = policz_wskazniki_procentowe(sezony)

    print()
    for nazwa in WSKAZNIKI_PROCENTOWE:
        braki = sezony[nazwa].isna().sum()
        print(
            f"  {nazwa:40s} mediana {sezony[nazwa].median():5.1f}%   "
            f"NaN {braki:4d} ({100 * braki / len(sezony):4.1f}%)"
        )

    if przyciete:
        for nazwa, ile in przyciete.items():
            print(f"  przyciete do [0,100]: {nazwa} - {ile} wierszy")
    else:
        print("\n  przycinanie do [0,100]: wszystkie wartosci w zakresie")

    mecze_po_90 = sezony["minuty_sezon"] / 90
    licznikowe = [
        c
        for c in sezony.columns
        if c not in KOLUMNY_NIELICZNIKOWE and c not in WSKAZNIKI_PROCENTOWE
    ]
    # wszystkie kolumny naraz - dokladanie po jednej jest bardzo wolne w pandas
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
    print(
        f"  kontrola: gole/90 mediana {sezony['goals_p90'].median():.2f}, "
        f"maks {sezony['goals_p90'].max():.2f}"
    )
    print(f"  stan po etapie: {len(sezony)} par, {sezony.shape[1]} kolumn")
    return sezony
