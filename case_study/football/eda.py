"""
Analiza eksploracyjna zbioru modelowego (etap E3).

Kazda funkcja w tym module odpowiada na jedno pytanie z listy E3 w `04-plan.md`,
zapisuje figure do `reports/figures/eda/`, tabele do `reports/tables/eda/`
i zwraca slownik liczb, ktore trafiaja do raportu `docs/05-eda.md`. Dzieki temu
kazda liczba w raporcie ma zrodlo w uruchamialnym kodzie, zgodnie z zasada pracy
z `01-projekt.md`.

Uruchomienie calosci:

    python -m case_study.football.uruchom_eda

Zasada dostepu do zbioru C (D-14): cechy zbioru C sa tutaj analizowane swobodnie,
bo pomiar przesuniecia rozkladow cech jest dostepny w kazdym wdrozeniu bez etykiet.
Zmienna celu zbioru C pojawia sie wylacznie jako liczby juz zapisane w `02-dane.md`.
Zbior testowy sluzy tu tylko do pomiaru przesuniecia cech - kazda ocena jakosci
modelu w tym module liczy sie na zbiorze kalibracyjnym.
"""

from __future__ import annotations

import matplotlib

# Backend "Agg" rysuje do pliku i nie probuje otwierac okna. Ustawienie musi
# poprzedzac import pyplot, bo backend wybiera sie raz, przy inicjalizacji.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .parametry import KATALOG_REPO, KATALOG_WYJSCIA, WSKAZNIKI_PROCENTOWE

# --- Sciezki wyjsciowe ---

KATALOG_FIGUR = KATALOG_REPO / "reports" / "figures" / "eda"
KATALOG_TABEL = KATALOG_REPO / "reports" / "tables" / "eda"


# --- Parametry analizy ---

# Progi, przy ktorych sprawdzamy redundancje cech. Filtr korelacyjny w E5 dostanie
# jedna z tych wartosci, wiec liczymy koszt kazdej.
PROGI_KORELACJI = (0.99, 0.95, 0.90)

# Prog, dla ktorego mierzymy, ile filtr korelacyjny faktycznie kupuje na jakosci.
PROG_FILTRA_DO_POMIARU = 0.95

# Prog laczenia cech w bloki przy grupowaniu hierarchicznym. Blok oznacza
# zestaw cech o sredniej korelacji bezwzglednej co najmniej tej wartosci.
PROG_BLOKU_CECH = 0.5

# Progi minut do analizy wrazliwosci. Wartosc 225 jest progiem obowiazujacym
# (D-08), wiec ponizej niej dane w zbiorze modelowym juz nie istnieja.
PROGI_MINUT = (225, 450, 900, 1350)

# Liczba koszy w indeksie PSI. Dziesiec decyli to konwencja z monitoringu modeli
# kredytowych, skad ta miara pochodzi.
KOSZE_PSI = 10

# Progi interpretacyjne PSI, rowniez konwencjonalne.
PSI_UMIARKOWANY = 0.10
PSI_DUZY = 0.25

# Kolejnosc sezonow do wykresow. Sortowanie alfabetyczne daje tu poprawna
# kolejnosc chronologiczna, bo sezon zapisany jest jako "RRRR-RRRR".
POZYCJE = ("DEF", "MID", "FOR")


# --- Infrastruktura ---

# Slownik zbierajacy wszystkie liczby, ktore trafia do raportu. Klucz jest
# tekstowa nazwa faktu, wartosc - liczba albo krotki napis.
FAKTY: dict[str, object] = {}


def zapisz_fakt(klucz: str, wartosc) -> None:
    """
    Odklada jedna liczbe do wspolnego slownika faktow.

    Slownik jest zmienna modulowa, wiec funkcje analityczne moga do niego dopisywac
    bez przekazywania go w argumentach. Na koncu przebiegu zrzucamy go do JSON-a,
    zeby raport dalo sie zweryfikowac liczba po liczbie.
    """
    # numpy zwraca wlasne typy (np.float64), ktorych json nie umie zapisac.
    # `item()` wyciaga z nich zwykly float albo int Pythona.
    if hasattr(wartosc, "item"):
        wartosc = wartosc.item()
    FAKTY[klucz] = wartosc


def ustaw_styl() -> None:
    """
    Wspolny styl figur. Wywolywane raz, na poczatku przebiegu.

    `rcParams` to globalny slownik ustawien matplotlib - zmiana tutaj dziala
    na wszystkie pozniej tworzone wykresy.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )


def zapisz_figure(fig, nazwa: str) -> None:
    """Zapisuje figure pod `reports/figures/eda/<nazwa>.png` i zwalnia pamiec."""
    KATALOG_FIGUR.mkdir(parents=True, exist_ok=True)
    fig.savefig(KATALOG_FIGUR / f"{nazwa}.png")
    # Bez close matplotlib trzyma kazda figure w pamieci do konca procesu.
    plt.close(fig)
    print(f"    figura  reports/figures/eda/{nazwa}.png")


def zapisz_tabele(tabela: pd.DataFrame, nazwa: str) -> None:
    """Zapisuje tabele pod `reports/tables/eda/<nazwa>.csv`."""
    KATALOG_TABEL.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(KATALOG_TABEL / f"{nazwa}.csv", index=True)
    print(f"    tabela   reports/tables/eda/{nazwa}.csv  ({len(tabela)} wierszy)")


def wczytaj() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wczytuje oba pliki z `data/processed/`.

    Zwraca:
        (model, sezony) - zbior modelowy oraz pelny zbior kontrolny z surowymi
        sumami sezonowymi. Ten drugi jest potrzebny wszedzie tam, gdzie pytamy
        o liczniki i mianowniki wskaznikow albo o date wyceny.
    """
    model = pd.read_parquet(KATALOG_WYJSCIA / "model.parquet")
    sezony = pd.read_parquet(KATALOG_WYJSCIA / "zawodnik_sezon.parquet")
    return model, sezony


def kolumny_cech(model: pd.DataFrame) -> dict[str, list[str]]:
    """
    Dzieli kolumny zbioru na bloki tematyczne.

    Bloki wracaja wielokrotnie - przy korelacjach, przy dryfie i przy sufcie
    informacyjnym - wiec wyznaczamy je raz, w jednym miejscu.

    Wskazniki procentowe bierzemy z `WSKAZNIKI_PROCENTOWE`, a nie z dopasowania
    nazwy do wzorca. Kolumna `tackled_perecentage` ma literowke w schemacie
    zrodlowym, wiec `endswith("percentage")` znajduje siedem wskaznikow zamiast
    osmiu i gubi akurat ten, ktory ma najwiecej brakow. Slownik z `parametry.py`
    jest lista rozstrzygajaca.
    """
    procentowe = [k for k in WSKAZNIKI_PROCENTOWE if k in model.columns]
    return {
        "p90": [k for k in model.columns if k.endswith("_p90")],
        "procentowe": procentowe,
        "liczbowe": ["wiek", "wiek_do_kw", "wzrost_cm", "minuty_sezon", "mecze"],
        "kategoryczne": ["pozycja", "region", "liga", "noga", "zmienil_lige"],
    }


def big5(model: pd.DataFrame) -> pd.DataFrame:
    """
    Zwraca same wiersze Big 5, czyli wszystko poza zbiorem C.

    Wiekszosc analiz celu dotyczy wylacznie Big 5, bo cel zbioru C zostaje
    zamkniety do eksperymentu koncowego (D-14).
    """
    return model[model["podzial"] != "zbior_C"]


# --- A1  rozklad zmiennej celu ---


def a1_rozklad_celu(model: pd.DataFrame) -> None:
    """
    Rozklad celu w trzech wersjach: nominalnej, logarytmicznej i zdeflowanej.

    Odpowiada na pierwszy punkt E3. Sprawdzamy symetrie po log1p, ktora jest
    uzasadnieniem D-03, oraz ziarnistosc siatki wycen Transfermarktu.
    """
    print("\nA1  rozklad zmiennej celu")
    b = big5(model)

    for nazwa, seria in [
        ("nominalny", b["market_value_in_eur"]),
        ("log1p", b["log_market_value"]),
        ("log1p_realny", b["log_market_value_real"]),
    ]:
        zapisz_fakt(f"a1_skosnosc_{nazwa}", round(float(seria.skew()), 3))
        zapisz_fakt(f"a1_kurtoza_{nazwa}", round(float(seria.kurtosis()), 3))
        print(
            f"    {nazwa:14s} skosnosc {seria.skew():+7.3f}   "
            f"kurtoza {seria.kurtosis():+7.3f}"
        )

    # Ziarnistosc celu. `value_counts(normalize=True)` daje udzialy zamiast liczb,
    # a `cumsum` po posortowaniu malejaco mowi, ile progow zbiera zadany udzial masy.
    udzialy = b["market_value_in_eur"].value_counts(normalize=True)
    skumulowane = udzialy.sort_values(ascending=False).cumsum()
    zapisz_fakt("a1_unikalnych_progow", int(b["market_value_in_eur"].nunique()))
    zapisz_fakt("a1_udzial_top10_progow", round(float(udzialy.head(10).sum()) * 100, 1))
    zapisz_fakt("a1_progow_na_polowe_masy", int((skumulowane < 0.5).sum() + 1))
    zapisz_fakt("a1_progow_na_90_masy", int((skumulowane < 0.9).sum() + 1))
    print(
        f"    unikalnych wartosci celu: {b['market_value_in_eur'].nunique()}, "
        f"top 10 progow zbiera {100 * udzialy.head(10).sum():.1f}% wierszy"
    )

    zapisz_fakt("a1_min_eur", int(b["market_value_in_eur"].min()))
    zapisz_fakt("a1_max_eur", int(b["market_value_in_eur"].max()))
    zapisz_fakt("a1_mediana_eur", int(b["market_value_in_eur"].median()))

    fig, osie = plt.subplots(1, 3, figsize=(12, 3.4))
    osie[0].hist(b["market_value_in_eur"] / 1e6, bins=60, color="#4C72B0")
    osie[0].set(
        title=f"Nominalny, skośność {b['market_value_in_eur'].skew():.2f}",
        xlabel="wartość [mln EUR]",
        ylabel="liczba par zawodnik-sezon",
    )
    # Skala logarytmiczna na osi Y, bo ogon nominalny jest dlugi i plaski.
    osie[0].set_yscale("log")

    osie[1].hist(b["log_market_value"], bins=60, color="#55A868")
    osie[1].set(
        title=f"log1p, skośność {b['log_market_value'].skew():.2f}",
        xlabel="log1p(wartość)",
    )

    osie[2].hist(b["log_market_value_real"], bins=60, color="#C44E52")
    osie[2].set(
        title=f"log1p zdeflowany, skośność {b['log_market_value_real'].skew():.2f}",
        xlabel="log1p(wartość realna)",
    )
    fig.suptitle("A1. Rozkład zmiennej celu, Big 5", y=1.04)
    zapisz_figure(fig, "a1_rozklad_celu")

    # Osobna figura na ziarnistosc: dwadziescia najczestszych progow.
    fig, os_ = plt.subplots(figsize=(7, 3.6))
    naj = udzialy.head(20).sort_index()
    os_.bar(range(len(naj)), naj.to_numpy() * 100, color="#4C72B0")
    os_.set_xticks(range(len(naj)))
    os_.set_xticklabels([f"{w / 1e6:g}" for w in naj.index], rotation=90)
    os_.set(
        title="A1. Dwadzieścia najczęstszych progów wyceny",
        xlabel="próg [mln EUR]",
        ylabel="udział par [%]",
    )
    zapisz_figure(fig, "a1_progi_wyceny")
    zapisz_tabele(udzialy.head(30).rename("udzial").to_frame(), "a1_progi_wyceny")


# --- A2  skutecznosc deflacji ---


def a2_deflacja(model: pd.DataFrame) -> None:
    """
    Czy po zdeflowaniu poziom celu jest plaski w czasie.

    Test jest prosty: liczymy srednia log-wartosci w kazdym sezonie, osobno dla
    celu nominalnego i realnego, i porownujemy rozstepy. Indeks liczony jest
    wylacznie na treningu (etap 9), wiec plaskosc obowiazuje z gory na sezonach
    treningowych. To, co zostaje poza treningiem, jest cena przedluzenia
    ostatniego wspolczynnika na sezony pozniejsze.
    """
    print("\nA2  skutecznosc deflacji")
    b = big5(model)

    poziom = b.groupby("season")[["log_market_value", "log_market_value_real"]].mean()
    poziom["mediana_nom_mln"] = (
        b.groupby("season")["market_value_in_eur"].median() / 1e6
    )
    poziom["mediana_real_mln"] = b.groupby("season")["market_value_real"].median() / 1e6
    poziom["par"] = b.groupby("season").size()
    print(poziom.round(3).to_string())

    sezony_treningowe = sorted(
        model.loc[model["podzial"] == "trening", "season"].unique()
    )
    trening = poziom.loc[sezony_treningowe]

    def rozstep(seria: pd.Series) -> float:
        """Roznica miedzy najwyzszym a najnizszym poziomem sezonowym."""
        return float(seria.max() - seria.min())

    zapisz_fakt(
        "a2_rozstep_nominalny_wszystkie", round(rozstep(poziom["log_market_value"]), 3)
    )
    zapisz_fakt(
        "a2_rozstep_realny_wszystkie",
        round(rozstep(poziom["log_market_value_real"]), 3),
    )
    zapisz_fakt(
        "a2_rozstep_nominalny_trening", round(rozstep(trening["log_market_value"]), 3)
    )
    zapisz_fakt(
        "a2_rozstep_realny_trening",
        round(rozstep(trening["log_market_value_real"]), 3),
    )
    zapisz_fakt(
        "a2_redukcja_na_treningu_razy",
        round(
            rozstep(trening["log_market_value"])
            / rozstep(trening["log_market_value_real"]),
            1,
        ),
    )
    zapisz_fakt(
        "a2_inflacja_nominalna_proc",
        round(
            100
            * (
                np.exp(
                    poziom["log_market_value"].iloc[-1]
                    - poziom["log_market_value"].iloc[0]
                )
                - 1
            ),
            1,
        ),
    )
    print(
        f"    rozstep sredniej log: nominalnie "
        f"{rozstep(poziom['log_market_value']):.3f}, "
        f"realnie {rozstep(poziom['log_market_value_real']):.3f}"
    )
    print(
        f"    na samych sezonach treningowych: "
        f"{rozstep(trening['log_market_value']):.3f} -> "
        f"{rozstep(trening['log_market_value_real']):.3f}"
    )

    fig, os_ = plt.subplots(figsize=(7.5, 4))
    x = range(len(poziom))
    os_.plot(
        x, poziom["log_market_value"], "o-", label="cel nominalny", color="#4C72B0"
    )
    os_.plot(
        x,
        poziom["log_market_value_real"],
        "s-",
        label="cel zdeflowany",
        color="#C44E52",
    )
    # Pionowa linia w miejscu, gdzie konczy sie trening i zaczyna plaskie
    # przedluzenie wspolczynnika deflacji.
    os_.axvline(len(sezony_treningowe) - 0.5, color="black", linestyle=":", linewidth=1)
    os_.text(
        len(sezony_treningowe) - 0.45,
        os_.get_ylim()[0],
        " koniec treningu,\n dalej płaskie przedłużenie",
        fontsize=7.5,
        va="bottom",
    )
    os_.set_xticks(list(x))
    os_.set_xticklabels(poziom.index, rotation=20)
    os_.set(
        title="A2. Poziom celu w czasie przed deflacją i po niej",
        ylabel="średnia log1p(wartość)",
    )
    os_.legend()
    zapisz_figure(fig, "a2_deflacja")
    zapisz_tabele(poziom.round(4), "a2_poziom_w_czasie")


# --- A3  krzywa wiek-wartosc ---


def a3_krzywa_wieku(model: pd.DataFrame) -> None:
    """
    Ksztalt zaleznosci wiek - wartosc oraz uzasadnienie cechy `wiek_do_kw`.

    Plan E3 stawia pytanie wprost: jesli ksztalt odbiega od paraboli, cecha
    `wiek_do_kw` jest ozdobnikiem i trzeba to napisac. Mierzymy to liczbowo,
    porownujac trzy dopasowania na tych samych danych:
      - prosta, czyli sam wiek,
      - parabola, czyli wiek i wiek do kwadratu,
      - srednia w kazdym roczniku, czyli model nieparametryczny wyznaczajacy
        sufit osiagalny dowolna funkcja samego wieku.
    Iloraz R2 paraboli i R2 sufitu mowi, jaka czesc informacji o wieku parabola
    faktycznie zbiera.
    """
    print("\nA3  krzywa wiek-wartosc")
    b = big5(model)
    x, y = b["wiek"].to_numpy(), b["log_market_value"].to_numpy()

    def r2(prognoza: np.ndarray) -> float:
        """Udzial wariancji celu wyjasniony przez podana prognoze."""
        return float(1 - np.var(y - prognoza) / np.var(y))

    # `np.polyfit(x, y, stopien)` dopasowuje wielomian metoda najmniejszych
    # kwadratow i zwraca wspolczynniki od najwyzszej potegi. `np.polyval` liczy
    # z nich wartosci wielomianu w zadanych punktach.
    prosta = np.polyfit(x, y, 1)
    parabola = np.polyfit(x, y, 2)
    # `transform("mean")` podstawia kazdemu wierszowi srednia jego grupy
    # i zachowuje dlugosc serii, wiec wychodzi gotowa prognoza nieparametryczna.
    sufit = b.groupby("wiek")["log_market_value"].transform("mean").to_numpy()

    r2_prosta = r2(np.polyval(prosta, x))
    r2_parabola = r2(np.polyval(parabola, x))
    r2_sufit = r2(sufit)
    # Wierzcholek paraboli a*w^2 + b*w + c lezy w punkcie -b / (2a).
    wierzcholek = float(-parabola[1] / (2 * parabola[0]))

    zapisz_fakt("a3_r2_prosta", round(r2_prosta, 4))
    zapisz_fakt("a3_r2_parabola", round(r2_parabola, 4))
    zapisz_fakt("a3_r2_sufit_rocznikowy", round(r2_sufit, 4))
    zapisz_fakt("a3_udzial_paraboli_w_sufcie", round(100 * r2_parabola / r2_sufit, 1))
    zapisz_fakt("a3_wierzcholek_wiek", round(wierzcholek, 1))
    print(
        f"    R2: prosta {r2_prosta:.4f} | parabola {r2_parabola:.4f} | "
        f"sufit rocznikowy {r2_sufit:.4f}"
    )
    print(
        f"    parabola zbiera {100 * r2_parabola / r2_sufit:.1f}% informacji "
        f"o wieku, wierzcholek w {wierzcholek:.1f} roku zycia"
    )

    rocznik = b.groupby("wiek").agg(
        par=("log_market_value", "size"),
        mediana_log=("log_market_value", "median"),
        srednia_log=("log_market_value", "mean"),
        mediana_mln=("market_value_in_eur", lambda s: s.median() / 1e6),
        mediana_minut=("minuty_sezon", "median"),
    )
    zapisz_tabele(rocznik.round(3), "a3_krzywa_wieku")

    # Najmlodsi zawodnicy sa w tym zbiorze drozsi niz zawodnicy w szczycie
    # kariery. Prog 225 minut przepuszcza z rocznika osiemnastolatkow wylacznie
    # tych, ktorzy dostaja gre w Big 5, czyli talenty juz wycenione wysoko.
    liczni = rocznik[rocznik["par"] >= 30]
    for wiek in (18, 25):
        zapisz_fakt(
            f"a3_mediana_mln_wiek_{wiek}",
            round(float(rocznik.loc[wiek, "mediana_mln"]), 2),
        )
        zapisz_fakt(f"a3_par_wiek_{wiek}", int(rocznik.loc[wiek, "par"]))
        zapisz_fakt(
            f"a3_mediana_minut_wiek_{wiek}", int(rocznik.loc[wiek, "mediana_minut"])
        )
    zapisz_fakt("a3_najdrozszy_rocznik", int(liczni["mediana_log"].idxmax()))

    fig, osie = plt.subplots(1, 2, figsize=(12, 4))
    siatka = np.linspace(x.min(), x.max(), 200)
    osie[0].scatter(
        rocznik.index,
        rocznik["mediana_log"],
        s=rocznik["par"] / 8,
        color="#4C72B0",
        label="mediana rocznika, pole = liczebność",
    )
    osie[0].plot(
        siatka,
        np.polyval(parabola, siatka),
        color="#C44E52",
        label=f"parabola, wierzchołek {wierzcholek:.1f}",
    )
    osie[0].plot(
        siatka,
        np.polyval(prosta, siatka),
        color="#8172B2",
        linestyle="--",
        label="prosta",
    )
    osie[0].set(
        title="A3. Wiek a wartość, Big 5",
        xlabel="wiek na 30 czerwca",
        ylabel="log1p(wartość)",
    )
    osie[0].legend(fontsize=7.5)

    for poz, kolor in zip(POZYCJE, ["#4C72B0", "#55A868", "#C44E52"]):
        s = b[b["pozycja"] == poz]
        krzywa = s.groupby("wiek")["log_market_value"].agg(["median", "size"])
        krzywa = krzywa[krzywa["size"] >= 20]
        osie[1].plot(
            krzywa.index, krzywa["median"], "o-", color=kolor, markersize=3.5, label=poz
        )
        wsp = np.polyfit(s["wiek"], s["log_market_value"], 2)
        zapisz_fakt(f"a3_wierzcholek_{poz}", round(float(-wsp[1] / (2 * wsp[0])), 1))
    osie[1].set(
        title="A3. Krzywa wieku w podziale na pozycje", xlabel="wiek na 30 czerwca"
    )
    osie[1].legend()
    zapisz_figure(fig, "a3_krzywa_wieku")


# --- A4  struktura panelu ---


def a4_struktura_panelu(model: pd.DataFrame) -> None:
    """
    Ilu zawodnikow w ilu sezonach, czyli wejscie do GroupKFold w E4.

    Grupowanie po `player_id` ma sens wtedy, gdy grup jest duzo wiecej niz foldow,
    a jednoczesnie czesc zawodnikow powtarza sie na tyle czesto, ze podzial losowy
    faktycznie rozrzucilby ich po foldach. Obie liczby sa tutaj.

    Przy okazji mierzymy lepkosc wyceny rok do roku. Cecha `market_value_prev`
    zostala odcieta modelom (D-05), a sama sila tej zaleznosci opisuje zadanie
    i uzasadnia baseline M0c z D-06.
    """
    print("\nA4  struktura panelu")

    sezonow_na_zawodnika = model.groupby("player_id")["season"].nunique()
    rozklad = sezonow_na_zawodnika.value_counts().sort_index()
    print("    liczba sezonow -> liczba zawodnikow:")
    print("      " + rozklad.to_string().replace("\n", "\n      "))

    zapisz_fakt("a4_zawodnikow", len(sezonow_na_zawodnika))
    zapisz_fakt(
        "a4_udzial_jednosezonowych_zawodnikow",
        round(100 * float((sezonow_na_zawodnika == 1).mean()), 1),
    )
    zapisz_fakt(
        "a4_udzial_jednosezonowych_wierszy",
        round(100 * int((sezonow_na_zawodnika == 1).sum()) / len(model), 1),
    )
    zapisz_fakt("a4_srednio_sezonow", round(float(sezonow_na_zawodnika.mean()), 2))
    zapisz_fakt("a4_wszystkie_szesc_sezonow", int((sezonow_na_zawodnika == 6).sum()))

    trening = model[model["podzial"] == "trening"]
    grupy_treningowe = trening["player_id"].nunique()
    zapisz_fakt("a4_grup_w_treningu", int(grupy_treningowe))
    zapisz_fakt(
        "a4_wierszy_na_grupe_trening", round(len(trening) / grupy_treningowe, 2)
    )
    print(
        f"    trening: {len(trening)} wierszy w {grupy_treningowe} grupach "
        f"({len(trening) / grupy_treningowe:.2f} wiersza na zawodnika)"
    )

    # Lepkosc wyceny. Numerujemy sezony rosnaco, kopie tabeli przesuwamy o jeden
    # sezon do przodu i laczymy - zostaja pary (zawodnik, sezon t) z dolaczona
    # wycena z sezonu t-1.
    kolejnosc = {s: i for i, s in enumerate(sorted(model["season"].unique()))}
    biezacy = model[["player_id", "season", "market_value_in_eur", "podzial"]].copy()
    biezacy["t"] = biezacy["season"].map(kolejnosc)
    poprzedni = biezacy[["player_id", "t", "market_value_in_eur"]].copy()
    poprzedni["t"] = poprzedni["t"] + 1
    pary = biezacy.merge(
        poprzedni,
        on=["player_id", "t"],
        suffixes=("", "_prev"),
        validate="one_to_one",
    )

    identyczne = float(
        (pary["market_value_in_eur"] == pary["market_value_in_eur_prev"]).mean()
    )
    korelacja = float(
        np.corrcoef(
            np.log(pary["market_value_in_eur"]),
            np.log(pary["market_value_in_eur_prev"]),
        )[0, 1]
    )
    zapisz_fakt("a4_par_z_poprzednia_wycena", len(pary))
    zapisz_fakt("a4_udzial_identycznych_wycen", round(100 * identyczne, 1))
    zapisz_fakt("a4_korelacja_log_z_poprzednim", round(korelacja, 4))
    zapisz_fakt("a4_r2_przepisania", round(korelacja**2, 4))

    test = model[model["podzial"] == "test"]
    pokrycie_test = len(pary[pary["podzial"] == "test"]) / len(test)
    zapisz_fakt("a4_pokrycie_t1_na_tescie", round(100 * pokrycie_test, 1))
    print(
        f"    par z wycena z t-1: {len(pary)}, identyczna wycena rok do roku: "
        f"{100 * identyczne:.1f}%, korelacja log: {korelacja:.4f}"
    )
    print(f"    pokrycie baseline M0c na tescie: {100 * pokrycie_test:.1f}%")

    fig, osie = plt.subplots(1, 2, figsize=(11.5, 3.8))
    osie[0].bar(rozklad.index, rozklad.to_numpy(), color="#4C72B0")
    osie[0].set(
        title="A4. Ilu zawodników w ilu sezonach",
        xlabel="liczba sezonów w zbiorze",
        ylabel="liczba zawodników",
    )
    for i, v in zip(rozklad.index, rozklad.to_numpy()):
        osie[0].text(i, v, str(v), ha="center", va="bottom", fontsize=7.5)

    osie[1].scatter(
        np.log1p(pary["market_value_in_eur_prev"]),
        np.log1p(pary["market_value_in_eur"]),
        s=3,
        alpha=0.15,
        color="#4C72B0",
    )
    granice = [
        float(np.log1p(pary["market_value_in_eur"]).min()),
        float(np.log1p(pary["market_value_in_eur"]).max()),
    ]
    osie[1].plot(granice, granice, color="#C44E52", linewidth=1)
    osie[1].set(
        title=f"A4. Wycena rok do roku, korelacja {korelacja:.3f}",
        xlabel="log1p(wartość w sezonie t−1)",
        ylabel="log1p(wartość w sezonie t)",
    )
    zapisz_figure(fig, "a4_struktura_panelu")
    zapisz_tabele(rozklad.rename("zawodnikow").to_frame(), "a4_sezonow_na_zawodnika")


# --- A5  cel i cechy w podgrupach ---


def a5_podgrupy(model: pd.DataFrame) -> None:
    """
    Rozklad celu w podgrupach oraz premia grupowa po kontroli obserwowalnych.

    Sama mediana w grupie miesza dwie rzeczy: sklad grupy i wycene grupy. Region
    AMERYKA_PLD ma inny rozklad wieku i inny rozklad lig niz EUROPA, wiec roznica
    surowych median moze pochodzic w calosci ze skladu. Dlatego liczymy tez
    premie warunkowa: dopasowujemy regresje liniowa na wieku, kwadracie wieku,
    pozycji, lidze i sezonie, a potem usredniamy reszty w kazdej grupie. To, co
    zostaje, jest czescia niewyjasniona przez te kontrole.

    Wynik jest wstepna diagnoza W5 (sprawiedliwosc) postawiona przed
    jakimkolwiek modelem - opisuje obciazenie wycen rynku, a nie modelu.
    """
    print("\nA5  cel i cechy w podgrupach")
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder

    b = big5(model).copy()
    kontrole_kat = ["pozycja", "liga", "season"]
    kontrole_num = ["wiek", "wiek_do_kw"]

    # `sparse_output=False` kaze enkoderowi zwrocic zwykla tablice numpy zamiast
    # macierzy rzadkiej, co upraszcza sklejenie z kolumnami liczbowymi.
    enkoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def premia(kolumna: str) -> pd.DataFrame:
        """
        Srednia reszta w grupie po odjeciu wplywu wieku, pozycji, ligi i sezonu.

        Analizowana zmienna wypada ze zbioru kontroli. Gdyby zostala, model
        kontrolny wyjasnilby jej wplyw w calosci i kazda premia wyszlaby rowna
        zeru z samej konstrukcji.
        """
        d = b[b[kolumna].notna()]
        kategoryczne = [k for k in kontrole_kat if k != kolumna]
        # `np.hstack` skleja tablice w poziomie, czyli dokleja kolumny.
        macierz = np.hstack(
            [enkoder.fit_transform(d[kategoryczne]), d[kontrole_num].to_numpy()]
        )
        model_kontrolny = LinearRegression().fit(macierz, d["log_market_value"])
        reszta = d["log_market_value"] - model_kontrolny.predict(macierz)
        wynik = reszta.groupby(d[kolumna]).agg(["size", "mean"])
        wynik.columns = ["par", "srednia_reszta"]
        # Reszta jest w skali logarytmicznej, wiec exp zamienia ja na mnoznik.
        wynik["mnoznik"] = np.exp(wynik["srednia_reszta"])
        return wynik

    surowe = []
    for kolumna in ["pozycja", "liga", "region", "noga", "zmienil_lige"]:
        opis = b.groupby(kolumna).agg(
            par=("log_market_value", "size"),
            mediana_mln=("market_value_in_eur", lambda s: s.median() / 1e6),
            srednia_log=("log_market_value", "mean"),
            odchylenie_log=("log_market_value", "std"),
            mediana_wieku=("wiek", "median"),
        )
        opis = opis.join(premia(kolumna)[["srednia_reszta", "mnoznik"]])
        opis.insert(0, "zmienna", kolumna)
        opis.index.name = "grupa"
        surowe.append(opis)
        print(f"    --- {kolumna} ---")
        print("      " + opis.round(3).to_string().replace("\n", "\n      "))

    tabela = pd.concat(surowe)
    zapisz_tabele(tabela.round(4), "a5_podgrupy")

    # Fakty do raportu: najwieksze premie warunkowe.
    for grupa in ["AMERYKA_PLD", "EUROPA", "RESZTA"]:
        w = tabela[(tabela["zmienna"] == "region")].loc[grupa]
        zapisz_fakt(f"a5_mnoznik_region_{grupa}", round(float(w["mnoznik"]), 3))
        zapisz_fakt(f"a5_par_region_{grupa}", int(w["par"]))
    zapisz_fakt(
        "a5_mnoznik_obie_nogi",
        round(float(tabela[tabela["zmienna"] == "noga"].loc["both", "mnoznik"]), 3),
    )
    zapisz_fakt(
        "a5_mnoznik_zmienil_lige",
        round(
            float(tabela[tabela["zmienna"] == "zmienil_lige"].loc[True, "mnoznik"]), 3
        ),
    )
    zapisz_fakt(
        "a5_par_zmienil_lige",
        int(tabela[tabela["zmienna"] == "zmienil_lige"].loc[True, "par"]),
    )

    # Rozrzut wewnatrz ligi wobec roznic miedzy ligami. Jesli odchylenie w lidze
    # jest duzo wieksze niz rozstep median, to liga sama w sobie tlumaczy malo.
    ligi = tabela[tabela["zmienna"] == "liga"]
    zapisz_fakt(
        "a5_rozstep_srednich_lig",
        round(float(ligi["srednia_log"].max() - ligi["srednia_log"].min()), 3),
    )
    zapisz_fakt(
        "a5_mediana_odchylenia_w_lidze",
        round(float(ligi["odchylenie_log"].median()), 3),
    )

    fig, osie = plt.subplots(1, 3, figsize=(13, 3.9))
    for os_, kolumna, tytul in zip(
        osie,
        ["pozycja", "liga", "region"],
        ["Pozycja", "Liga", "Region"],
    ):
        grupy = [g for g, _ in b.groupby(kolumna)]
        dane = [b.loc[b[kolumna] == g, "log_market_value"].to_numpy() for g in grupy]
        os_.boxplot(dane, tick_labels=grupy, showfliers=False)
        os_.set(
            title=f"A5. {tytul}",
            ylabel="log1p(wartość)" if kolumna == "pozycja" else None,
        )
        os_.tick_params(axis="x", rotation=45 if kolumna == "liga" else 0)
    fig.suptitle("A5. Rozkład celu w podgrupach, Big 5", y=1.03)
    zapisz_figure(fig, "a5_podgrupy")

    # Druga figura: premia warunkowa jako mnoznik.
    fig, os_ = plt.subplots(figsize=(7.5, 3.6))
    wybrane = tabela[tabela["zmienna"].isin(["region", "noga", "zmienil_lige"])]
    etykiety = [f"{z}: {g}" for z, g in zip(wybrane["zmienna"], wybrane.index)]
    kolory = ["#C44E52" if m > 1 else "#4C72B0" for m in wybrane["mnoznik"]]
    os_.barh(etykiety, (wybrane["mnoznik"] - 1) * 100, color=kolory)
    os_.axvline(0, color="black", linewidth=0.8)
    os_.set(
        title="A5. Premia warunkowa po kontroli wieku, pozycji, ligi i sezonu",
        xlabel="odchylenie od oczekiwanej wyceny [%]",
    )
    zapisz_figure(fig, "a5_premia_warunkowa")


def _koszt_redundancji(
    model: pd.DataFrame, cechy: list[str], prog: float
) -> dict[str, float]:
    """
    Porownuje jakosc modelu liniowego bez filtra korelacyjnego i w dwoch jego
    wariantach.

    Uczymy na treningu, oceniamy na kalibracji, wszedzie ten sam potok i ta sama
    siatka alf. Jedyna roznica to lista cech ciaglych podanych na wejsciu, wiec
    cala roznica w wyniku pochodzi z filtra.

    Warianty roznia sie regula wyboru cechy do usuniecia z pary:

    - naiwny: zostaje cecha stojaca w parze pierwsza, czyli wczesniejsza
      alfabetycznie. Tak dziala filtr napisany najprosciej, jak sie da;
    - swiadomy celu: zostaje cecha silniej skorelowana ze zmienna celu, przy
      czym pary przetwarzamy od najsilniej skorelowanej. Korelacje z celem
      liczymy wylacznie na treningu, wiec regula nie zaglada poza niego.

    Przyjmuje:
        model - caly zbior modelowy
        cechy - lista cech ciaglych do przefiltrowania
        prog  - prog korelacji bezwzglednej, powyzej ktorego para idzie do odsiewu

    Zwraca:
        slownik z R2 trzech wariantow oraz liczba cech, ktore zostaja.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    bloki = kolumny_cech(model)
    trening = model[model["podzial"] == "trening"]
    kalibracja = model[model["podzial"] == "kalibracja"]

    macierz = trening[cechy].corr().abs()
    maska = np.triu(np.ones(macierz.shape, dtype=bool), 1)
    pary = macierz.where(maska).stack()

    usuniete_naiwnie: set[str] = set()
    for a, c in pary[pary > prog].index:
        if a not in usuniete_naiwnie:
            usuniete_naiwnie.add(c)

    # `corrwith` liczy korelacje kazdej kolumny z podana seria; bierzemy modul,
    # bo interesuje nas sila zwiazku niezaleznie od kierunku.
    sila_zwiazku = trening[cechy].corrwith(trening["log_market_value"]).abs()
    usuniete_swiadomie: set[str] = set()
    # `sort_values(ascending=False)` przetwarza pary od najsilniej skorelowanej,
    # wiec najbardziej oczywiste duplikaty rozstrzygamy w pierwszej kolejnosci.
    for a, c in pary[pary > prog].sort_values(ascending=False).index:
        if a in usuniete_swiadomie or c in usuniete_swiadomie:
            continue
        usuniete_swiadomie.add(c if sila_zwiazku[a] >= sila_zwiazku[c] else a)

    zostaje_naiwnie = [k for k in cechy if k not in usuniete_naiwnie]
    zostaje_swiadomie = [k for k in cechy if k not in usuniete_swiadomie]

    def ocen(lista_cech: list[str]) -> float:
        """Uczy potok na treningu i zwraca R2 na kalibracji."""
        num = bloki["liczbowe"] + lista_cech
        kat = bloki["kategoryczne"]
        przygotowanie = ColumnTransformer(
            [
                (
                    "liczbowe",
                    Pipeline(
                        [
                            ("imputacja", SimpleImputer(strategy="median")),
                            ("skala", StandardScaler()),
                        ]
                    ),
                    num,
                ),
                ("kategoryczne", OneHotEncoder(handle_unknown="ignore"), kat),
            ]
        )
        potok = Pipeline(
            [
                ("przygotowanie", przygotowanie),
                ("model", RidgeCV(alphas=np.logspace(-2, 4, 25))),
            ]
        )
        potok.fit(trening[num + kat], trening["log_market_value"])
        return float(
            r2_score(
                kalibracja["log_market_value"], potok.predict(kalibracja[num + kat])
            )
        )

    bez_filtra = ocen(cechy)
    filtr_naiwny = ocen(zostaje_naiwnie)
    filtr_swiadomy = ocen(zostaje_swiadomie)
    return {
        "r2_bez_filtra": round(bez_filtra, 4),
        "r2_filtr_naiwny": round(filtr_naiwny, 4),
        "r2_filtr_swiadomy_celu": round(filtr_swiadomy, 4),
        "zysk_filtra_naiwnego": round(filtr_naiwny - bez_filtra, 4),
        "zysk_filtra_swiadomego": round(filtr_swiadomy - bez_filtra, 4),
        "cech_po_filtrze": len(zostaje_naiwnie),
    }


# --- A6  korelacje miedzy cechami ---


def a6_korelacje(model: pd.DataFrame) -> None:
    """
    Ile z cech `_p90` jest redundantnych, czyli wejscie do filtra korelacyjnego w E5.

    Korelacje liczymy wylacznie na zbiorze treningowym, bo filtr w Pipeline
    bedzie fitowany na treningu i to jego decyzje odtwarzamy tutaj.

    Symulujemy dokladnie ta procedure, ktora zastosuje filtr: idziemy po parach
    o korelacji powyzej progu i z kazdej takiej pary usuwamy druga cecha,
    zachowujac pierwsza. Wynikiem jest liczba cech, ktore filtr usunie.
    """
    print("\nA6  korelacje miedzy cechami")
    bloki = kolumny_cech(model)
    cechy = bloki["p90"] + bloki["procentowe"]
    trening = model[model["podzial"] == "trening"]

    macierz = trening[cechy].corr().abs()
    # `np.triu(..., 1)` daje maske gornego trojkata bez przekatnej, czyli kazda
    # pare cech dokladnie raz. `where` zostawia tylko te pola, `stack` splaszcza
    # tabele do serii indeksowanej para nazw.
    maska = np.triu(np.ones(macierz.shape, dtype=bool), 1)
    pary = macierz.where(maska).stack()

    for prog in PROGI_KORELACJI:
        zapisz_fakt(f"a6_par_powyzej_{prog}", int((pary > prog).sum()))
        # Symulacja filtra: usuwamy druga cecha z kazdej pary powyzej progu.
        do_usuniecia: set[str] = set()
        # Indeks serii `pary` jest dwupoziomowy, wiec kazdy jego element
        # rozpakowuje sie na nazwy obu cech tworzacych pare.
        for a, c in pary[pary > prog].index:
            if a not in do_usuniecia:
                do_usuniecia.add(c)
        zapisz_fakt(f"a6_cech_usunietych_przy_{prog}", len(do_usuniecia))
        print(
            f"    prog {prog}: {int((pary > prog).sum()):3d} par, "
            f"filtr usuwa {len(do_usuniecia):2d} z {len(cechy)} cech"
        )

    najsilniejsze = pary.sort_values(ascending=False).head(25)
    tabela = najsilniejsze.rename("korelacja").reset_index()
    tabela.columns = ["cecha_a", "cecha_b", "korelacja"]
    zapisz_tabele(tabela.round(4), "a6_pary_skorelowane")
    print("    dziesiec najsilniejszych par:")
    print(
        "      "
        + tabela.head(10).round(4).to_string(index=False).replace("\n", "\n      ")
    )

    zapisz_fakt("a6_liczba_cech_ciaglych", len(cechy))
    zapisz_fakt("a6_najsilniejsza_para", f"{tabela.iloc[0, 0]} / {tabela.iloc[0, 1]}")
    zapisz_fakt("a6_najsilniejsza_korelacja", round(float(tabela.iloc[0, 2]), 5))
    zapisz_fakt("a6_druga_para", f"{tabela.iloc[1, 0]} / {tabela.iloc[1, 1]}")
    zapisz_fakt("a6_druga_korelacja", round(float(tabela.iloc[1, 2]), 5))
    # Mediana korelacji bezwzglednej mowi, jak gesta jest cala macierz.
    zapisz_fakt("a6_mediana_korelacji", round(float(pary.median()), 4))

    # Ile filtr faktycznie kupuje. Model liniowy z regularyzacja grzbietowa jest
    # tu wrazliwym probnikiem: para cech skorelowanych niemal doskonale pozwala
    # dopasowac duze wspolczynniki o przeciwnych znakach, ktorych roznica jest
    # na treningu prawie stala. Gdy poziom takiej cechy przesunie sie poza
    # treningiem, blad rosnie proporcjonalnie do tych wspolczynnikow.
    koszt = _koszt_redundancji(model, cechy, PROG_FILTRA_DO_POMIARU)
    for nazwa, wartosc in koszt.items():
        zapisz_fakt(f"a6_{nazwa}", wartosc)
    print(
        f"    Ridge na kalibracji, prog {PROG_FILTRA_DO_POMIARU}: "
        f"bez filtra R2 {koszt['r2_bez_filtra']:+.4f} | "
        f"filtr naiwny {koszt['r2_filtr_naiwny']:+.4f} "
        f"({koszt['zysk_filtra_naiwnego']:+.4f}) | "
        f"filtr swiadomy celu {koszt['r2_filtr_swiadomy_celu']:+.4f} "
        f"({koszt['zysk_filtra_swiadomego']:+.4f})"
    )

    # Struktura blokowa i efektywna wymiarowosc.
    #
    # Sama macierz korelacji 95 na 95 jest nieczytelna, wiec zadajemy jej dwa
    # konkretne pytania. Po pierwsze: w ile grup wzajemnie zamiennych cech
    # rozpada sie ten zestaw. Po drugie: ilu niezaleznych wymiarow te 95 kolumn
    # naprawde dostarcza. Odpowiedz na oba pytania mowi, ile realnie mierzymy.
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    odleglosc = 1 - macierz.to_numpy()
    # Macierz odleglosci musi byc dokladnie symetryczna i miec zera na przekatnej,
    # inaczej `squareform` odmawia. Drobne bledy zaokraglen usuwamy jawnie.
    np.fill_diagonal(odleglosc, 0.0)
    odleglosc = (odleglosc + odleglosc.T) / 2
    # `squareform` zamienia kwadratowa macierz odleglosci na wektor, ktorego
    # oczekuje `linkage`. Metoda "average" laczy grupy po sredniej odleglosci
    # miedzy ich elementami, wiec blok oznacza tu zbior cech wzajemnie
    # skorelowanych, a nie lancuch par.
    powiazania = hierarchy.linkage(
        squareform(odleglosc, checks=False), method="average"
    )
    porzadek = hierarchy.leaves_list(powiazania)
    uporzadkowana = macierz.iloc[porzadek, porzadek]

    # `fcluster` z kryterium "distance" tnie drzewo na wysokosci 1 - prog, czyli
    # zostawia w jednym bloku cechy o sredniej korelacji bezwzglednej co najmniej
    # rownej progowi.
    klastry = hierarchy.fcluster(
        powiazania, t=1 - PROG_BLOKU_CECH, criterion="distance"
    )
    klastry_w_porzadku = klastry[porzadek]
    nazwy_w_porzadku = [cechy[i] for i in porzadek]

    # Granice blokow to miejsca, w ktorych numer klastra sie zmienia.
    granice = [
        i
        for i in range(1, len(klastry_w_porzadku))
        if klastry_w_porzadku[i] != klastry_w_porzadku[i - 1]
    ]
    poczatki = [0, *granice]
    konce = [*granice, len(klastry_w_porzadku)]

    opis_blokow = []
    for numer, (od_, do_) in enumerate(zip(poczatki, konce), start=1):
        czlonkowie = nazwy_w_porzadku[od_:do_]
        # Reprezentantem bloku jest cecha najsilniej zwiazana z pozostalymi
        # w tym samym bloku, czyli jego srodek.
        wewnatrz = macierz.loc[czlonkowie, czlonkowie]
        reprezentant = (
            czlonkowie[0] if len(czlonkowie) == 1 else wewnatrz.mean().idxmax()
        )
        opis_blokow.append(
            {
                "blok": numer,
                "cech": len(czlonkowie),
                "reprezentant": reprezentant,
                "srednia_korelacja_w_bloku": round(
                    float(
                        wewnatrz.where(~np.eye(len(czlonkowie), dtype=bool))
                        .stack()
                        .mean()
                    )
                    if len(czlonkowie) > 1
                    else 1.0,
                    3,
                ),
                "czlonkowie": ", ".join(czlonkowie),
            }
        )

    tabela_blokow = pd.DataFrame(opis_blokow).set_index("blok")
    zapisz_tabele(tabela_blokow, "a6_bloki_cech")
    zapisz_fakt("a6_blokow", len(tabela_blokow))
    zapisz_fakt("a6_najwiekszy_blok", int(tabela_blokow["cech"].max()))
    zapisz_fakt(
        "a6_reprezentant_najwiekszego_bloku",
        str(tabela_blokow.loc[tabela_blokow["cech"].idxmax(), "reprezentant"]),
    )
    zapisz_fakt("a6_blokow_jednoelementowych", int((tabela_blokow["cech"] == 1).sum()))
    print(
        f"    przy |r| >= {PROG_BLOKU_CECH}: {len(tabela_blokow)} blokow, "
        f"najwiekszy ma {tabela_blokow['cech'].max()} cech "
        f"({tabela_blokow.loc[tabela_blokow['cech'].idxmax(), 'reprezentant']})"
    )

    # Efektywna wymiarowosc. Wartosci wlasne macierzy kowariancji cech
    # standaryzowanych mowia, ile wariancji niesie kazdy kolejny niezalezny
    # kierunek. Liczba kierunkow potrzebnych na 80 i 90 procent wariancji jest
    # uczciwa miara tego, ile niezaleznych pomiarow tu w ogole mamy.
    dane = trening[cechy]
    # Braki uzupelniamy mediana treningu wylacznie na potrzeby tego pomiaru.
    dane = dane.fillna(dane.median())
    standaryzowane = (dane - dane.mean()) / dane.std()
    # `eigvalsh` liczy wartosci wlasne macierzy symetrycznej i zwraca je rosnaco,
    # wiec odwracamy kolejnosc przez [::-1].
    wartosci_wlasne = np.linalg.eigvalsh(np.cov(standaryzowane.to_numpy().T))[::-1]
    skumulowana = np.cumsum(wartosci_wlasne) / wartosci_wlasne.sum()
    skladowych_80 = int((skumulowana < 0.80).sum() + 1)
    skladowych_90 = int((skumulowana < 0.90).sum() + 1)
    # Wymiarowosc partycypacyjna: (suma wartosci wlasnych)^2 / suma kwadratow.
    # Dla zestawu k cech idealnie nieskorelowanych wynosi k, a dla zestawu
    # sprowadzajacego sie do jednego wymiaru spada do jednosci.
    wymiarowosc = float(wartosci_wlasne.sum() ** 2 / np.sum(wartosci_wlasne**2))
    zapisz_fakt("a6_skladowych_na_80_proc", skladowych_80)
    zapisz_fakt("a6_skladowych_na_90_proc", skladowych_90)
    zapisz_fakt("a6_wymiarowosc_partycypacyjna", round(wymiarowosc, 1))
    zapisz_fakt(
        "a6_pierwsza_skladowa_proc",
        round(100 * float(wartosci_wlasne[0] / wartosci_wlasne.sum()), 1),
    )
    print(
        f"    efektywna wymiarowosc: {skladowych_80} skladowych na 80% wariancji, "
        f"{skladowych_90} na 90%, pierwsza tlumaczy "
        f"{100 * wartosci_wlasne[0] / wartosci_wlasne.sum():.1f}%"
    )

    fig, osie = plt.subplots(
        1, 2, figsize=(13.5, 6.2), gridspec_kw={"width_ratios": [1.25, 1]}
    )

    obraz = osie[0].imshow(uporzadkowana, cmap="magma", vmin=0, vmax=1)
    # Bloki obrysowujemy ramka zamiast ciac obraz liniami przez cala szerokosc.
    # Przy 28 blokach, z ktorych polowa jest jednoelementowa, siatka linii
    # zaciemnialaby to, co ma pokazac.
    etykiety_pozycje, etykiety_tekst = [], []
    for wiersz, (od_, do_) in zip(opis_blokow, zip(poczatki, konce)):
        if wiersz["cech"] < 3:
            continue
        osie[0].add_patch(
            plt.Rectangle(
                (od_ - 0.5, od_ - 0.5),
                wiersz["cech"],
                wiersz["cech"],
                fill=False,
                edgecolor="#7FDBFF",
                linewidth=1.0,
            )
        )
        etykiety_pozycje.append((od_ + do_ - 1) / 2)
        etykiety_tekst.append(f"{wiersz['reprezentant']}  ({wiersz['cech']})")
    osie[0].set_yticks(etykiety_pozycje)
    osie[0].set_yticklabels(etykiety_tekst, fontsize=6.5)
    osie[0].set_xticks([])
    osie[0].set_title(
        f"A6. {len(cechy)} cech układa się w {len(tabela_blokow)} bloków\n"
        f"(próg |r| ≥ {PROG_BLOKU_CECH}; podpisane bloki od trzech cech wzwyż)",
        fontsize=9.5,
    )
    fig.colorbar(obraz, ax=osie[0], shrink=0.75, label="|r|")

    osie[1].plot(
        range(1, len(skumulowana) + 1), 100 * skumulowana, "-", color="#4C72B0"
    )
    for prog_wariancji, liczba, kolor in [
        (80, skladowych_80, "#55A868"),
        (90, skladowych_90, "#C44E52"),
    ]:
        osie[1].axhline(prog_wariancji, color=kolor, linewidth=0.8, linestyle=":")
        osie[1].axvline(liczba, color=kolor, linewidth=0.8, linestyle=":")
        osie[1].plot(liczba, prog_wariancji, "o", color=kolor)
        osie[1].annotate(
            f"{liczba} składowych na {prog_wariancji}%",
            (liczba, prog_wariancji),
            textcoords="offset points",
            xytext=(8, -12),
            fontsize=8,
            color=kolor,
        )
    osie[1].set(
        title=f"A6. Ile tu jest niezależnej informacji\n"
        f"(wymiarowość partycypacyjna {wymiarowosc:.1f})",
        xlabel="liczba składowych głównych",
        ylabel="skumulowana wariancja cech [%]",
        ylim=(0, 101),
    )
    zapisz_figure(fig, "a6_korelacje")


# --- A7  mapa brakow ---


def a7_mapa_brakow(model: pd.DataFrame) -> None:
    """
    Czy braki rozkladaja sie losowo, czy koncentruja w jednej grupie.

    Pytanie ma konsekwencje dla W5: imputacja mediana w cechach, ktorych braki
    siedza prawie wylacznie u obroncow, podstawi obroncom wartosc typowa dla
    calej populacji i wprowadzi obciazenie grupowe.

    Test chi-kwadrat sprawdza, czy udzial brakow rozni sie miedzy pozycjami
    bardziej, niz wynikaloby to z losowego rozrzutu.
    """
    print("\nA7  mapa brakow")
    kolumny_z_brakami = [k for k in model.columns if model[k].isna().any()]
    wiersze = []

    for kolumna in kolumny_z_brakami:
        brak = model[kolumna].isna()
        # `crosstab` buduje tabele licznosci pozycja x (brak / obecny).
        tabela_kontyngencji = pd.crosstab(model["pozycja"], brak)
        # `chi2_contingency` zwraca statystyke, p-wartosc, stopnie swobody
        # i tabele oczekiwana; interesuje nas druga pozycja.
        p_wartosc = float(stats.chi2_contingency(tabela_kontyngencji)[1])
        udzial = brak.groupby(model["pozycja"]).mean() * 100
        wiersze.append(
            {
                "cecha": kolumna,
                "brakow": int(brak.sum()),
                "udzial_proc": round(100 * float(brak.mean()), 2),
                "DEF_proc": round(float(udzial.get("DEF", 0)), 2),
                "MID_proc": round(float(udzial.get("MID", 0)), 2),
                "FOR_proc": round(float(udzial.get("FOR", 0)), 2),
                "chi2_p": p_wartosc,
                "mediana_minut_brak": float(model.loc[brak, "minuty_sezon"].median()),
                "mediana_minut_reszta": float(
                    model.loc[~brak, "minuty_sezon"].median()
                ),
            }
        )

    tabela = pd.DataFrame(wiersze).sort_values("brakow", ascending=False)
    print("      " + tabela.to_string(index=False).replace("\n", "\n      "))
    zapisz_tabele(tabela.set_index("cecha"), "a7_mapa_brakow")

    glowna = tabela[tabela["cecha"] == "dribble_success_percentage"].iloc[0]
    zapisz_fakt("a7_braki_dribble", int(glowna["brakow"]))
    zapisz_fakt("a7_dribble_DEF_proc", float(glowna["DEF_proc"]))
    zapisz_fakt("a7_dribble_FOR_proc", float(glowna["FOR_proc"]))
    zapisz_fakt(
        "a7_dribble_iloraz_DEF_FOR",
        round(float(glowna["DEF_proc"] / glowna["FOR_proc"]), 1),
    )
    zapisz_fakt("a7_dribble_minuty_brak", int(glowna["mediana_minut_brak"]))
    zapisz_fakt("a7_dribble_minuty_reszta", int(glowna["mediana_minut_reszta"]))
    zapisz_fakt("a7_kolumn_z_brakami", len(kolumny_z_brakami))
    zapisz_fakt("a7_cech_z_istotna_koncentracja", int((tabela["chi2_p"] < 0.01).sum()))

    fig, osie = plt.subplots(1, 2, figsize=(12, 3.8))
    wykres = tabela.set_index("cecha")[["DEF_proc", "MID_proc", "FOR_proc"]]
    wykres.plot(kind="barh", ax=osie[0], color=["#4C72B0", "#55A868", "#C44E52"])
    osie[0].set(
        title="A7. Udział braków według pozycji", xlabel="braki [%]", ylabel=None
    )
    osie[0].legend(["DEF", "MID", "FOR"], fontsize=8)
    osie[0].tick_params(axis="y", labelsize=7)

    brak = model["dribble_success_percentage"].isna()
    osie[1].hist(
        [model.loc[brak, "minuty_sezon"], model.loc[~brak, "minuty_sezon"]],
        bins=25,
        density=True,
        color=["#C44E52", "#4C72B0"],
        label=["brak wskaźnika", "wskaźnik obecny"],
    )
    osie[1].set(
        title="A7. Braki a wolumen gry (dribble_success_percentage)",
        xlabel="minuty w sezonie",
        ylabel="gęstość",
    )
    osie[1].legend(fontsize=8)
    zapisz_figure(fig, "a7_mapa_brakow")


# --- A8  wrazliwosc na prog minut ---


def a8_prog_minut(model: pd.DataFrame) -> None:
    """
    Jak prog minut zmienia liczebnosc, sklad i wariancje cech.

    Ograniczenie do odnotowania: zbior modelowy jest juz odfiltrowany progiem 225
    (D-08), wiec warianty ponizej tej wartosci wymagalyby ponownego przebiegu
    pipeline z innym `MIN_MINUT`. Tutaj mierzymy kierunek i sile efektu na
    progach od 225 w gore, co wystarcza do oceny, czy prog jest parametrem
    obojetnym dla wnioskow.

    Wariancja cech `_p90` mierzona jest wspolczynnikiem zmiennosci, czyli
    odchyleniem podzielonym przez srednia. Jest bezwymiarowy, wiec da sie go
    usrednic po cechach o roznych jednostkach.
    """
    print("\nA8  wrazliwosc na prog minut")
    b = big5(model)
    p90 = kolumny_cech(model)["p90"]
    wiersze = []

    for prog in PROGI_MINUT:
        podzbior = b[b["minuty_sezon"] >= prog]
        # Wspolczynnik zmiennosci liczony cecha po cesze, potem mediana po cechach.
        zmiennosc = (podzbior[p90].std() / podzbior[p90].mean().abs()).median()
        wiersze.append(
            {
                "prog": prog,
                "par": len(podzbior),
                "udzial_proc": round(100 * len(podzbior) / len(b), 1),
                "zawodnikow": podzbior["player_id"].nunique(),
                "mediana_wieku": float(podzbior["wiek"].median()),
                "udzial_do_21_proc": round(
                    100 * float((podzbior["wiek"] <= 21).mean()), 1
                ),
                "mediana_mln": round(
                    float(podzbior["market_value_in_eur"].median()) / 1e6, 2
                ),
                "srednia_log": round(float(podzbior["log_market_value"].mean()), 3),
                "mediana_wsp_zmiennosci": round(float(zmiennosc), 4),
                "braki_dribble_proc": round(
                    100 * float(podzbior["dribble_success_percentage"].isna().mean()), 2
                ),
            }
        )

    tabela = pd.DataFrame(wiersze).set_index("prog")
    print("      " + tabela.to_string().replace("\n", "\n      "))
    zapisz_tabele(tabela, "a8_prog_minut")

    zapisz_fakt("a8_udzial_przy_450", float(tabela.loc[450, "udzial_proc"]))
    zapisz_fakt("a8_udzial_przy_900", float(tabela.loc[900, "udzial_proc"]))
    zapisz_fakt("a8_mlodzi_przy_225", float(tabela.loc[225, "udzial_do_21_proc"]))
    zapisz_fakt("a8_mlodzi_przy_900", float(tabela.loc[900, "udzial_do_21_proc"]))
    zapisz_fakt("a8_mediana_mln_225", float(tabela.loc[225, "mediana_mln"]))
    zapisz_fakt("a8_mediana_mln_900", float(tabela.loc[900, "mediana_mln"]))
    zapisz_fakt(
        "a8_spadek_zmiennosci_proc",
        round(
            100
            * (
                1
                - tabela.loc[900, "mediana_wsp_zmiennosci"]
                / tabela.loc[225, "mediana_wsp_zmiennosci"]
            ),
            1,
        ),
    )

    fig, osie = plt.subplots(1, 3, figsize=(12.5, 3.5))
    osie[0].plot(tabela.index, tabela["udzial_proc"], "o-", color="#4C72B0")
    osie[0].set(title="A8. Ile par zostaje", xlabel="próg minut", ylabel="udział [%]")
    osie[1].plot(tabela.index, tabela["udzial_do_21_proc"], "o-", color="#C44E52")
    osie[1].set(
        title="A8. Udział zawodników do 21 lat", xlabel="próg minut", ylabel="[%]"
    )
    osie[2].plot(tabela.index, tabela["mediana_wsp_zmiennosci"], "o-", color="#55A868")
    osie[2].set(
        title="A8. Zmienność cech _p90",
        xlabel="próg minut",
        ylabel="mediana współczynnika zmienności",
    )
    zapisz_figure(fig, "a8_prog_minut")


# --- A9  sufit informacyjny ---


def a9_sufit_informacyjny(model: pd.DataFrame) -> None:
    """
    Ile wariancji celu tlumaczy sam kontekst, zanim dolozymy statystyki gry.

    Budujemy zagniezdzone bloki cech i po kazdym dolozeniu mierzymy R2. Rosnaca
    roznica miedzy blokami mowi, ile wnosi kazda porcja informacji, a ostatni
    wiersz podaje realistyczne oczekiwanie wobec modeli z E5.

    Ocena idzie na zbiorze kalibracyjnym. Zbior testowy zostaje nietkniety przez
    caly etap E3, zeby protokol z E4 mogl go uzyc jako danych naprawde
    niewidzianych. Kalibracja ma tu role tymczasowego zbioru oceny; w E5 wroci
    do swojej wlasciwej roli z W3.
    """
    print("\nA9  sufit informacyjny")
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    bloki = kolumny_cech(model)
    trening = model[model["podzial"] == "trening"]
    kalibracja = model[model["podzial"] == "kalibracja"]
    y_tr, y_ka = trening["log_market_value"], kalibracja["log_market_value"]

    def ocen(num: list[str], kat: list[str], nazwa: str, estymator=None) -> dict:
        """
        Uczy prosty potok na treningu i ocenia go na kalibracji.

        ColumnTransformer przepuszcza kolumny liczbowe przez imputacje mediana
        i standaryzacje, a kategoryczne przez kodowanie zerojedynkowe. Cala
        obrobka siedzi wewnatrz Pipeline, wiec fituje sie wylacznie na treningu -
        dokladnie tak, jak wymaga D-09.
        """
        kroki = []
        if num:
            kroki.append(
                (
                    "liczbowe",
                    Pipeline(
                        [
                            ("imputacja", SimpleImputer(strategy="median")),
                            ("skala", StandardScaler()),
                        ]
                    ),
                    num,
                )
            )
        if kat:
            kroki.append(("kategoryczne", OneHotEncoder(handle_unknown="ignore"), kat))
        # `np.logspace(-2, 4, 25)` daje 25 wartosci alfa rozlozonych rownomiernie
        # w skali logarytmicznej od 0,01 do 10 000; RidgeCV wybiera z nich
        # najlepsza walidacja krzyzowa na treningu.
        est = estymator or RidgeCV(alphas=np.logspace(-2, 4, 25))
        potok = Pipeline([("przygotowanie", ColumnTransformer(kroki)), ("model", est)])
        potok.fit(trening[num + kat], y_tr)
        prognoza = potok.predict(kalibracja[num + kat])
        wynik = {
            "blok": nazwa,
            "cech": len(num) + len(kat),
            "r2": round(float(r2_score(y_ka, prognoza)), 4),
            "mae_log": round(float(mean_absolute_error(y_ka, prognoza)), 4),
        }
        print(f"    {nazwa:46s} cech {wynik['cech']:3d}   R2 {wynik['r2']:+.4f}")
        return wynik

    wszystkie_num = bloki["liczbowe"] + bloki["p90"] + bloki["procentowe"]
    wszystkie_kat = bloki["kategoryczne"]

    # Mediana treningu jako punkt zerowy - odpowiednik baseline M0a z D-06.
    mediana = float(y_tr.median())
    wiersze = [
        {
            "blok": "M0a mediana globalna",
            "cech": 0,
            "r2": round(float(r2_score(y_ka, np.full(len(y_ka), mediana))), 4),
            "mae_log": round(
                float(mean_absolute_error(y_ka, np.full(len(y_ka), mediana))), 4
            ),
        }
    ]
    print(f"    {'M0a mediana globalna':46s} cech   0   R2 {wiersze[0]['r2']:+.4f}")

    wiersze.append(ocen(["wiek", "wiek_do_kw"], [], "wiek i kwadrat wieku"))
    wiersze.append(
        ocen(["wiek", "wiek_do_kw"], ["pozycja", "liga"], "+ pozycja i liga")
    )
    wiersze.append(
        ocen(
            ["wiek", "wiek_do_kw"],
            ["pozycja", "liga", "region", "noga"],
            "+ region i noga",
        )
    )
    wiersze.append(
        ocen(
            ["wiek", "wiek_do_kw", "wzrost_cm", "minuty_sezon", "mecze"],
            ["pozycja", "liga", "region", "noga"],
            "+ wolumen gry (minuty, mecze) i wzrost",
        )
    )
    wiersze.append(
        ocen(wszystkie_num, wszystkie_kat, "+ wszystkie statystyki gry (Ridge)")
    )
    wiersze.append(
        ocen(
            wszystkie_num,
            wszystkie_kat,
            "to samo, HistGradientBoosting",
            HistGradientBoostingRegressor(random_state=0),
        )
    )
    # Kolejnosc odwrocona: same statystyki gry, bez kontekstu i bez wolumenu.
    wiersze.append(
        ocen(
            bloki["p90"] + bloki["procentowe"], [], "same statystyki gry, bez kontekstu"
        )
    )
    wiersze.append(ocen(["minuty_sezon", "mecze"], [], "sam wolumen gry"))

    tabela = pd.DataFrame(wiersze).set_index("blok")
    zapisz_tabele(tabela, "a9_sufit_informacyjny")

    def r2_bloku(nazwa: str) -> float:
        return float(tabela.loc[nazwa, "r2"])

    zapisz_fakt("a9_r2_sam_wiek", r2_bloku("wiek i kwadrat wieku"))
    zapisz_fakt("a9_r2_kontekst", r2_bloku("+ region i noga"))
    zapisz_fakt(
        "a9_r2_kontekst_wolumen", r2_bloku("+ wolumen gry (minuty, mecze) i wzrost")
    )
    zapisz_fakt("a9_r2_ridge_wszystko", r2_bloku("+ wszystkie statystyki gry (Ridge)"))
    zapisz_fakt("a9_r2_histgb_wszystko", r2_bloku("to samo, HistGradientBoosting"))
    zapisz_fakt("a9_r2_same_statystyki", r2_bloku("same statystyki gry, bez kontekstu"))
    zapisz_fakt("a9_r2_sam_wolumen", r2_bloku("sam wolumen gry"))
    zapisz_fakt(
        "a9_przyrost_od_wolumenu",
        round(
            r2_bloku("+ wolumen gry (minuty, mecze) i wzrost")
            - r2_bloku("+ region i noga"),
            4,
        ),
    )
    zapisz_fakt(
        "a9_przyrost_od_statystyk",
        round(
            r2_bloku("+ wszystkie statystyki gry (Ridge)")
            - r2_bloku("+ wolumen gry (minuty, mecze) i wzrost"),
            4,
        ),
    )

    fig, os_ = plt.subplots(figsize=(8.5, 4))
    kolejnosc = tabela.iloc[:7]
    os_.barh(range(len(kolejnosc)), kolejnosc["r2"], color="#4C72B0")
    os_.set_yticks(range(len(kolejnosc)))
    os_.set_yticklabels(kolejnosc.index, fontsize=8)
    os_.invert_yaxis()
    os_.axvline(0, color="black", linewidth=0.8)
    for i, v in enumerate(kolejnosc["r2"]):
        os_.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    os_.set(
        title="A9. Sufit informacyjny: R² na kalibracji po dołożeniu kolejnych bloków cech",
        xlabel="R² w skali logarytmicznej",
    )
    zapisz_figure(fig, "a9_sufit_informacyjny")


# --- A10  przesuniecie rozkladow cech ---


def psi(odniesienie: pd.Series, porownanie: pd.Series, kosze: int = KOSZE_PSI) -> float:
    """
    Population Stability Index miedzy dwoma rozkladami jednej cechy.

    Kosze wyznaczamy decylami rozkladu odniesienia, potem liczymy udzialy obu
    prob w kazdym koszu i sumujemy (udzial_b - udzial_a) * log(udzial_b / udzial_a).
    Miara jest symetryczna wzgledem zamiany znaku roznicy i rosnie wraz
    z rozjechaniem sie rozkladow. Konwencja: ponizej 0,1 stabilnie, 0,1-0,25
    umiarkowanie, powyzej 0,25 duzo.
    """
    krawedzie = np.nanpercentile(odniesienie, np.linspace(0, 100, kosze + 1))
    # Skrajne kosze rozciagamy do nieskonczonosci, zeby zlapac wartosci spoza
    # zakresu proby odniesienia.
    krawedzie[0], krawedzie[-1] = -np.inf, np.inf
    # Cechy o wielu powtarzajacych sie wartosciach daja zduplikowane krawedzie,
    # ktore `np.histogram` odrzuca. `np.unique` zostawia rosnaca, unikalna liste.
    krawedzie = np.unique(krawedzie)
    if len(krawedzie) < 3:
        return float("nan")
    udzial_a = (
        np.histogram(odniesienie.dropna(), krawedzie)[0] / odniesienie.notna().sum()
    )
    udzial_b = (
        np.histogram(porownanie.dropna(), krawedzie)[0] / porownanie.notna().sum()
    )
    # Pusty kosz dalby logarytm z zera, wiec podlogujemy udzialy mala stala.
    udzial_a = np.clip(udzial_a, 1e-4, None)
    udzial_b = np.clip(udzial_b, 1e-4, None)
    return float(np.sum((udzial_b - udzial_a) * np.log(udzial_b / udzial_a)))


def a10_dryf_cech(model: pd.DataFrame) -> None:
    """
    Przesuniecie rozkladow cech: test wobec zbioru C oraz trening wobec testu.

    Dwa pomiary obok siebie odpowiadaja na pytanie, ktore z przesuniec jest
    wieksze: przejscie do innej ligi czy sam uplyw czasu wewnatrz Big 5.
    Zgodnie z D-14 dotykamy wylacznie cech zbioru C; jego zmienna celu zostaje
    zamknieta do eksperymentu koncowego.
    """
    print("\nA10  przesuniecie rozkladow cech")
    bloki = kolumny_cech(model)
    cechy = (
        ["wiek", "wzrost_cm", "minuty_sezon", "mecze"]
        + bloki["p90"]
        + bloki["procentowe"]
    )

    trening = model[model["podzial"] == "trening"]
    test = model[model["podzial"] == "test"]
    zbior_c = model[model["podzial"] == "zbior_C"]

    wiersze = []
    for cecha in cechy:
        # `ks_2samp` porownuje dwa rozklady empiryczne; statystyka to najwieksza
        # pionowa odleglosc miedzy ich dystrybuantami, wiec mierzy w tych samych
        # jednostkach dla kazdej cechy.
        ks = stats.ks_2samp(test[cecha].dropna(), zbior_c[cecha].dropna())
        wiersze.append(
            {
                "cecha": cecha,
                "psi_test_vs_C": round(psi(test[cecha], zbior_c[cecha]), 4),
                "ks_test_vs_C": round(float(ks.statistic), 4),
                "ks_p": float(ks.pvalue),
                "psi_trening_vs_test": round(psi(trening[cecha], test[cecha]), 4),
                "srednia_test": round(float(test[cecha].mean()), 4),
                "srednia_C": round(float(zbior_c[cecha].mean()), 4),
            }
        )

    tabela = (
        pd.DataFrame(wiersze)
        .set_index("cecha")
        .sort_values("psi_test_vs_C", ascending=False)
    )
    zapisz_tabele(tabela, "a10_dryf_cech")
    print("    dziesiec cech o najwiekszym przesunieciu do zbioru C:")
    print(
        "      "
        + tabela.head(10)[["psi_test_vs_C", "ks_test_vs_C", "psi_trening_vs_test"]]
        .to_string()
        .replace("\n", "\n      ")
    )

    duzy = int((tabela["psi_test_vs_C"] > PSI_DUZY).sum())
    umiarkowany = int(
        (
            (tabela["psi_test_vs_C"] > PSI_UMIARKOWANY)
            & (tabela["psi_test_vs_C"] <= PSI_DUZY)
        ).sum()
    )
    zapisz_fakt("a10_cech_ocenionych", len(tabela))
    zapisz_fakt("a10_psi_duzy", duzy)
    zapisz_fakt("a10_psi_umiarkowany", umiarkowany)
    zapisz_fakt("a10_psi_stabilny", len(tabela) - duzy - umiarkowany)
    zapisz_fakt("a10_mediana_psi_C", round(float(tabela["psi_test_vs_C"].median()), 4))
    zapisz_fakt(
        "a10_mediana_psi_czas", round(float(tabela["psi_trening_vs_test"].median()), 4)
    )
    zapisz_fakt(
        "a10_cech_z_psi_czas_powyzej_umiarkowanego",
        int((tabela["psi_trening_vs_test"] > PSI_UMIARKOWANY).sum()),
    )
    zapisz_fakt("a10_najwiekszy_dryf_C", tabela.index[0])
    zapisz_fakt("a10_najwiekszy_psi_C", float(tabela["psi_test_vs_C"].iloc[0]))
    najwiekszy_czas = tabela["psi_trening_vs_test"].idxmax()
    zapisz_fakt("a10_najwiekszy_dryf_czas", najwiekszy_czas)
    zapisz_fakt(
        "a10_najwiekszy_psi_czas",
        float(tabela.loc[najwiekszy_czas, "psi_trening_vs_test"]),
    )
    print(
        f"    PSI wobec zbioru C: {duzy} cech duzo, {umiarkowany} umiarkowanie, "
        f"{len(tabela) - duzy - umiarkowany} stabilnie"
    )
    print(
        f"    mediana PSI: do zbioru C {tabela['psi_test_vs_C'].median():.4f}, "
        f"sam uplyw czasu {tabela['psi_trening_vs_test'].median():.4f}"
    )

    fig, osie = plt.subplots(1, 2, figsize=(12, 4))
    osie[0].scatter(
        tabela["psi_trening_vs_test"],
        tabela["psi_test_vs_C"],
        s=14,
        alpha=0.7,
        color="#4C72B0",
    )
    granica = float(
        max(tabela["psi_trening_vs_test"].max(), tabela["psi_test_vs_C"].max())
    )
    osie[0].plot(
        [0, granica], [0, granica], color="black", linewidth=0.8, linestyle=":"
    )
    osie[0].axhline(PSI_DUZY, color="#C44E52", linewidth=0.8)
    osie[0].axhline(PSI_UMIARKOWANY, color="#DD8452", linewidth=0.8)

    # Obie osie w tej samej skali, bo przerywana przekatna sluzy do porownania
    # jednego przesuniecia z drugim. Zakres wyznaczamy przed podpisami, bo od
    # niego zalezy strona, po ktorej podpis sie miesci.
    zakres = (-0.02, granica + 0.06)

    # Cechy o najwiekszym przesunieciu leza blisko siebie, wiec podpisy stawiane
    # wprost na punktach nachodza na siebie. Rozsuwamy je pionowo o stala liczbe
    # punktow typograficznych i laczymy z punktem cienka kreska; `textcoords`
    # z wartoscia "offset points" liczy przesuniecie w jednostkach rysunku,
    # niezaleznie od skali osi.
    do_podpisania = tabela.head(4).index.tolist() + [najwiekszy_czas]
    for numer, cecha in enumerate(do_podpisania):
        x = float(tabela.loc[cecha, "psi_trening_vs_test"])
        y = float(tabela.loc[cecha, "psi_test_vs_C"])
        # Punkt lezacy w prawej polowie wykresu dostaje podpis po lewej stronie,
        # inaczej tekst wyszedlby poza os i wszedl na sasiedni panel.
        w_prawej_polowie = x > (zakres[0] + zakres[1]) / 2
        osie[0].annotate(
            cecha,
            (x, y),
            textcoords="offset points",
            xytext=(-24, 12) if w_prawej_polowie else (24, 18 - 15 * numer),
            fontsize=6.5,
            ha="right" if w_prawej_polowie else "left",
            arrowprops={"arrowstyle": "-", "linewidth": 0.5, "color": "#888888"},
        )
    osie[0].set(
        title="A10. Dwa źródła przesunięcia rozkładów",
        xlabel="PSI: trening → test (sam czas, Big 5)",
        ylabel="PSI: test → zbiór C (inna liga)",
        xlim=zakres,
        ylim=zakres,
    )

    naj = tabela.head(15)["psi_test_vs_C"].sort_values()
    osie[1].barh(range(len(naj)), naj.to_numpy(), color="#C44E52")
    osie[1].set_yticks(range(len(naj)))
    osie[1].set_yticklabels(naj.index, fontsize=7)
    osie[1].axvline(PSI_DUZY, color="black", linewidth=0.8, linestyle=":")
    osie[1].set(title="A10. Największe przesunięcia do zbioru C", xlabel="PSI")
    zapisz_figure(fig, "a10_dryf_cech")


# --- A11  zlom definicyjny w zrodle ---


def a11_zlom_definicyjny(model: pd.DataFrame) -> None:
    """
    Cechy, ktorych poziom skacze miedzy sezonami mocniej, niz tlumaczy to gra.

    Kazda cecha dostaje srednia sezonowa, a te standaryzujemy jej wlasnym
    odchyleniem miedzy zawodnikami. Dzieki temu skoki roznych cech sa
    porownywalne. Cecha, ktorej srednia przesuwa sie miedzy dwoma sasiednimi
    sezonami o ulamek odchylenia populacyjnego, zmienila najpewniej definicje
    u dostawcy danych, bo sama gra nie zmienia sie tak szybko.

    Konsekwencja jest powazna: przy podziale czasowym model uczy sie cechy
    w jednej definicji, a ocenia w drugiej.
    """
    print("\nA11  zlom definicyjny w zrodle")
    bloki = kolumny_cech(model)
    cechy = bloki["p90"] + bloki["procentowe"]
    b = big5(model)
    sezony = sorted(b["season"].unique())

    srednie = b.groupby("season")[cechy].mean().loc[sezony]
    # Odchylenie miedzy zawodnikami, liczone na calym Big 5 - jednostka odniesienia.
    odchylenia = b[cechy].std()
    # `diff()` liczy roznice miedzy kolejnymi wierszami, czyli miedzy sezonami.
    skoki = (srednie.diff() / odchylenia).abs()

    # Rozrozniamy dwa ksztalty zmiany, bo maja rozne konsekwencje.
    #
    # Trwaly krok to roznica miedzy poziomem dwoch ostatnich a dwoch pierwszych
    # sezonow. Taki ksztalt pasuje do zmiany definicji u dostawcy: poziom
    # przesuwa sie raz i zostaje. Przy podziale czasowym model uczy sie wtedy
    # cechy w jednej definicji, a ocenia w drugiej.
    #
    # Anomalia jednosezonowa to odchylenie sezonu 2020-2021 od sredniej jego
    # dwoch sasiadow. Taki ksztalt pasuje do warunkow konkretnego sezonu; ten
    # akurat rozegrano przy pustych trybunach i przy scisnietym terminarzu.
    okres_wczesny = [sezony[0], sezony[1]]
    okres_pozny = [sezony[-2], sezony[-1]]
    sezon_pandemiczny = "2020-2021"
    sasiedzi = ["2018-2019", "2021-2022"]

    wiersze = []
    for cecha in cechy:
        najwiekszy = skoki[cecha].max()
        gdzie = skoki[cecha].idxmax()
        krok = (
            srednie.loc[okres_pozny, cecha].mean()
            - srednie.loc[okres_wczesny, cecha].mean()
        ) / odchylenia[cecha]
        anomalia = (
            srednie.loc[sezon_pandemiczny, cecha] - srednie.loc[sasiedzi, cecha].mean()
        ) / odchylenia[cecha]
        # Calkowita zmiana od pierwszego do ostatniego sezonu, w procentach.
        zmiana = 100 * (srednie[cecha].iloc[-1] / srednie[cecha].iloc[0] - 1)
        wiersze.append(
            {
                "cecha": cecha,
                "najwiekszy_skok_sd": round(float(najwiekszy), 4),
                "miedzy_sezonami": f"{sezony[sezony.index(gdzie) - 1]} -> {gdzie}",
                "trwaly_krok_sd": round(float(krok), 4),
                "anomalia_2020_2021_sd": round(float(anomalia), 4),
                "zmiana_calkowita_proc": round(float(zmiana), 1),
                "srednia_pierwszy_sezon": round(float(srednie[cecha].iloc[0]), 3),
                "srednia_ostatni_sezon": round(float(srednie[cecha].iloc[-1]), 3),
            }
        )

    tabela = (
        pd.DataFrame(wiersze)
        .set_index("cecha")
        .sort_values("najwiekszy_skok_sd", ascending=False)
    )
    zapisz_tabele(tabela, "a11_zlom_definicyjny")
    print("      " + tabela.head(8).to_string().replace("\n", "\n      "))

    czolo = tabela.iloc[0]
    zapisz_fakt("a11_najwiekszy_skok_cecha", tabela.index[0])
    zapisz_fakt("a11_najwiekszy_skok_sd", float(czolo["najwiekszy_skok_sd"]))
    zapisz_fakt("a11_najwiekszy_skok_gdzie", str(czolo["miedzy_sezonami"]))
    zapisz_fakt("a11_najwiekszy_skok_zmiana", float(czolo["zmiana_calkowita_proc"]))
    zapisz_fakt(
        "a11_cech_ze_skokiem_powyzej_02sd",
        int((tabela["najwiekszy_skok_sd"] > 0.2).sum()),
    )

    # Ile cech ma najwiekszy skok w sasiedztwie sezonu pandemicznego. `str.contains`
    # sprawdza obecnosc napisu w kazdym wierszu kolumny tekstowej.
    przy_pandemii = int(tabela["miedzy_sezonami"].str.contains(sezon_pandemiczny).sum())
    zapisz_fakt("a11_cech_ze_skokiem_przy_2020_2021", przy_pandemii)
    zapisz_fakt(
        "a11_udzial_skokow_przy_pandemii", round(100 * przy_pandemii / len(tabela), 1)
    )
    zapisz_fakt(
        "a11_cech_z_trwalym_krokiem_powyzej_02sd",
        int(tabela["trwaly_krok_sd"].abs().gt(0.2).sum()),
    )
    zapisz_fakt(
        "a11_cech_z_anomalia_pandemiczna_powyzej_02sd",
        int(tabela["anomalia_2020_2021_sd"].abs().gt(0.2).sum()),
    )
    najwiekszy_krok = tabela["trwaly_krok_sd"].abs().idxmax()
    zapisz_fakt("a11_najwiekszy_trwaly_krok_cecha", najwiekszy_krok)
    zapisz_fakt(
        "a11_najwiekszy_trwaly_krok_sd",
        float(tabela.loc[najwiekszy_krok, "trwaly_krok_sd"]),
    )
    najwieksza_anomalia = tabela["anomalia_2020_2021_sd"].abs().idxmax()
    zapisz_fakt("a11_najwieksza_anomalia_cecha", najwieksza_anomalia)
    zapisz_fakt(
        "a11_najwieksza_anomalia_sd",
        float(tabela.loc[najwieksza_anomalia, "anomalia_2020_2021_sd"]),
    )
    print(
        f"    skok w sasiedztwie 2020-2021: {przy_pandemii} z {len(tabela)} cech "
        f"({100 * przy_pandemii / len(tabela):.1f}%)"
    )
    print(
        f"    trwaly krok powyzej 0,2 sd: "
        f"{int(tabela['trwaly_krok_sd'].abs().gt(0.2).sum())} cech | "
        f"anomalia pandemiczna powyzej 0,2 sd: "
        f"{int(tabela['anomalia_2020_2021_sd'].abs().gt(0.2).sum())} cech"
    )

    # Dowod wprost dla najsilniejszego przypadku. Proba dryblingu ma dwa mozliwe
    # zakonczenia - powodzenie albo odbior - wiec oba wskazniki powinny sumowac
    # sie do stu. Sprawdzamy, czy ta tozsamosc obowiazuje w kazdym sezonie.
    # Jesli przestaje obowiazywac od konkretnego sezonu, to dostawca dolozyl
    # trzecia kategorie zakonczenia i zmienil tym samym znaczenie obu kolumn.
    para = b.dropna(subset=["dribble_success_percentage", "tackled_perecentage"])
    suma = para["dribble_success_percentage"] + para["tackled_perecentage"]
    tozsamosc = pd.DataFrame(
        {
            "srednia_sumy": suma.groupby(para["season"]).mean(),
            "udzial_sumy_100_proc": 100
            * suma.round(6).eq(100).groupby(para["season"]).mean(),
            "korelacja_pary": para.groupby("season")[
                ["dribble_success_percentage", "tackled_perecentage"]
            ]
            .corr()
            # `unstack` rozklada wielopoziomowy indeks na kolumny, a potem
            # wyciagamy jedna komorke macierzy korelacji dla kazdego sezonu.
            .unstack()["dribble_success_percentage"]["tackled_perecentage"],
        }
    )
    zapisz_tabele(tozsamosc.round(3), "a11_tozsamosc_dryblingu")
    print("    tozsamosc wskaznikow dryblingu w sezonach:")
    print("      " + tozsamosc.round(3).to_string().replace("\n", "\n      "))

    ostatni_trening = max(model.loc[model["podzial"] == "trening", "season"].unique())
    pierwszy_po = sezony[sezony.index(ostatni_trening) + 1]
    zapisz_fakt(
        "a11_tozsamosc_przed_proc",
        round(float(tozsamosc.loc[ostatni_trening, "udzial_sumy_100_proc"]), 1),
    )
    zapisz_fakt(
        "a11_tozsamosc_po_proc",
        round(float(tozsamosc.loc[pierwszy_po, "udzial_sumy_100_proc"]), 1),
    )
    zapisz_fakt(
        "a11_suma_wskaznikow_przed",
        round(float(tozsamosc.loc[ostatni_trening, "srednia_sumy"]), 2),
    )
    zapisz_fakt(
        "a11_suma_wskaznikow_po",
        round(float(tozsamosc.loc[pierwszy_po, "srednia_sumy"]), 2),
    )
    zapisz_fakt("a11_granica_zmiany_definicji", f"{ostatni_trening} -> {pierwszy_po}")

    # Trzy cechy o najwiekszym skoku plus dwie najstabilniejsze jako tlo
    # porownawcze. Cechy kontrolne wybieramy z danych, bo intuicyjny wybor
    # trafil na cechy, ktore same okazaly sie niestabilne.
    kontrolne = tabela.nsmallest(2, "najwiekszy_skok_sd").index.tolist()
    do_wykresu = tabela.head(3).index.tolist() + kontrolne
    fig, osie = plt.subplots(1, len(do_wykresu), figsize=(3 * len(do_wykresu), 3.2))
    for os_, cecha in zip(osie, do_wykresu):
        wartosci = srednie[cecha]
        os_.plot(range(len(wartosci)), wartosci.to_numpy(), "o-", color="#C44E52")
        os_.set_xticks(range(len(wartosci)))
        os_.set_xticklabels([s[2:4] + "/" + s[7:9] for s in wartosci.index], fontsize=7)
        os_.set_title(cecha, fontsize=8)
    fig.suptitle(
        "A11. Średnie sezonowe: trzy cechy o największym skoku i dwie najstabilniejsze",
        y=1.04,
    )
    zapisz_figure(fig, "a11_zlom_definicyjny")

    # Osobno: sumy surowe z pliku kontrolnego pokazuja, czy skok idzie z licznika,
    # czy z mianownika. Liczby doklada funkcja ciekawostek.


# --- A12  ciekawostki ---


def a12_ciekawostki(model: pd.DataFrame, sezony: pd.DataFrame) -> None:
    """
    Fakty nieoczywiste: mechanika zrodla, redundancje i zaskakujace zaleznosci.

    Kazdy z tych pomiarow wyszedl przy okazji innego pytania i zostaje tutaj,
    bo zmienia sposob czytania zbioru.
    """
    print("\nA12  ciekawostki")
    b = big5(model)
    bloki = kolumny_cech(model)

    # 1. Fale wycen. Transfermarkt aktualizuje wyceny zbiorowo, w kilku dniach
    # w roku, wiec "data wyceny" jest data akcji redakcyjnej.
    daty = pd.to_datetime(sezony["data_wyceny"])
    licznosc_dat = daty.dt.strftime("%Y-%m-%d").value_counts()
    zapisz_fakt("a12_unikalnych_dat_wyceny", int(licznosc_dat.size))
    zapisz_fakt(
        "a12_udzial_20_najczestszych_dat",
        round(100 * float(licznosc_dat.head(20).sum()) / len(daty), 1),
    )
    zapisz_fakt("a12_najliczniejsza_data", str(licznosc_dat.index[0]))
    zapisz_fakt("a12_wycen_w_najliczniejszym_dniu", int(licznosc_dat.iloc[0]))
    zapisz_fakt("a12_mediana_dni_od_30_06", float(sezony["dni_od_30_06"].median()))
    print(
        f"    fale wycen: 20 najczestszych dat zbiera "
        f"{100 * licznosc_dat.head(20).sum() / len(daty):.1f}% wszystkich wycen"
    )

    # 2. Para kolumn, ktora przetrwala odsiew redundancji z etapu 1. FBref
    # obiecuje roznice miedzy dotknieciami ogolem a dotknieciami przy pilce
    # w grze, a w danych ta roznica prawie nie istnieje.
    roznica = sezony["total_touches"] - sezony["live_ball_touches"]
    zapisz_fakt(
        "a12_touches_identycznych_proc", round(100 * float((roznica == 0).mean()), 1)
    )
    zapisz_fakt("a12_touches_srednia_roznica", round(float(roznica.mean()), 2))
    zapisz_fakt("a12_touches_max_roznica", int(roznica.max()))
    zapisz_fakt(
        "a12_touches_korelacja",
        round(float(b["total_touches_p90"].corr(b["live_ball_touches_p90"])), 6),
    )
    print(
        f"    total_touches i live_ball_touches: identyczne w "
        f"{100 * (roznica == 0).mean():.1f}% wierszy, srednia roznica "
        f"{roznica.mean():.2f} dotkniecia na sezon"
    )

    # 2b. Druga para bliznaczych kolumn, tym razem z definicji. Proba dryblingu
    # konczy sie albo powodzeniem, albo odbiorem, wiec oba wskazniki licza to samo
    # zdarzenie z dwoch stron i sumuja sie do stu. Zostaje niewielka reszta na
    # przypadki trzecie, na przyklad przerwanie akcji faulem.
    wskazniki = model[["dribble_success_percentage", "tackled_perecentage"]].dropna()
    # `sum(axis=1)` sumuje wzdluz wierszy, czyli dodaje obie kolumny do siebie.
    suma_wskaznikow = wskazniki.sum(axis=1)
    zapisz_fakt(
        "a12_drybling_suma_rowna_100_proc",
        round(100 * float((suma_wskaznikow.round(6) == 100).mean()), 1),
    )
    zapisz_fakt("a12_drybling_srednia_sumy", round(float(suma_wskaznikow.mean()), 2))
    zapisz_fakt(
        "a12_drybling_korelacja",
        round(
            float(
                model["dribble_success_percentage"].corr(model["tackled_perecentage"])
            ),
            5,
        ),
    )
    print(
        f"    dribble_success i tackled_perecentage sumuja sie do 100 w "
        f"{100 * (suma_wskaznikow.round(6) == 100).mean():.1f}% wierszy "
        f"(srednia sumy {suma_wskaznikow.mean():.2f})"
    )

    # 3. Symetria ksiegowa danych zdarzeniowych. Kazdy udany odbior przy
    # dryblingu jest jednoczesnie strata dla dryblujacego, wiec sumy obu kolumn
    # musza sie zgadzac na poziomie ligi, choc na poziomie zawodnika opisuja
    # zupelnie rozne osoby.
    symetria = sezony.groupby(["liga", "season"])[
        ["tackled", "dribblers_tackled"]
    ].sum()
    symetria["iloraz"] = symetria["dribblers_tackled"] / symetria["tackled"]
    zapisz_fakt(
        "a12_symetria_mediana_ilorazu", round(float(symetria["iloraz"].median()), 4)
    )
    zapisz_fakt(
        "a12_symetria_max_odchylenie_proc",
        round(100 * float((symetria["iloraz"] - 1).abs().max()), 2),
    )
    zapisz_fakt(
        "a12_symetria_korelacja_zawodnik",
        round(float(sezony["tackled"].corr(sezony["dribblers_tackled"])), 3),
    )
    print(
        f"    suma tackled wobec dribblers_tackled w lidze i sezonie: mediana ilorazu "
        f"{symetria['iloraz'].median():.4f}, korelacja na poziomie zawodnika "
        f"{sezony['tackled'].corr(sezony['dribblers_tackled']):.3f}"
    )

    # 4. Ranking korelacji cech z celem. Wolumen gry bije wszystkie statystyki
    # jakosciowe, co jest istotne dla interpretacji rankingow SHAP w W4.
    kandydaci = (
        bloki["p90"]
        + bloki["procentowe"]
        + ["minuty_sezon", "mecze", "wiek", "wzrost_cm"]
    )
    korelacje = (
        b[kandydaci].corrwith(b["log_market_value"]).sort_values(ascending=False)
    )
    zapisz_tabele(
        korelacje.rename("korelacja_z_celem").to_frame().round(4),
        "a12_korelacje_z_celem",
    )
    zapisz_fakt("a12_najsilniejsza_cecha", str(korelacje.index[0]))
    zapisz_fakt("a12_najsilniejsza_korelacja", round(float(korelacje.iloc[0]), 3))
    zapisz_fakt("a12_korelacja_minuty", round(float(korelacje["minuty_sezon"]), 3))
    zapisz_fakt(
        "a12_najsilniejsza_boiskowa",
        str(korelacje.drop(["minuty_sezon", "mecze"]).index[0]),
    )
    zapisz_fakt(
        "a12_najsilniejsza_boiskowa_korelacja",
        round(float(korelacje.drop(["minuty_sezon", "mecze"]).iloc[0]), 3),
    )
    zapisz_fakt("a12_najbardziej_ujemna", str(korelacje.index[-1]))
    zapisz_fakt("a12_najbardziej_ujemna_korelacja", round(float(korelacje.iloc[-1]), 3))
    print(
        f"    najsilniejsza korelacja z celem: {korelacje.index[0]} "
        f"({korelacje.iloc[0]:+.3f}), najsilniejsza boiskowa: "
        f"{korelacje.drop(['minuty_sezon', 'mecze']).index[0]} "
        f"({korelacje.drop(['minuty_sezon', 'mecze']).iloc[0]:+.3f})"
    )

    # 5. Wzrost. Globalnie korelacja z wartoscia jest zerowa, a wewnatrz pozycji
    # zmienia znak - przyklad efektu Simpsona na tym zbiorze.
    d = b[b["wzrost_cm"].notna()]
    zapisz_fakt(
        "a12_wzrost_korelacja_globalna",
        round(float(d["wzrost_cm"].corr(d["log_market_value"])), 3),
    )
    for poz in POZYCJE:
        s = d[d["pozycja"] == poz]
        zapisz_fakt(
            f"a12_wzrost_korelacja_{poz}",
            round(float(s["wzrost_cm"].corr(s["log_market_value"])), 3),
        )
        zapisz_fakt(f"a12_wzrost_mediana_{poz}", float(s["wzrost_cm"].median()))
    print(
        f"    wzrost: korelacja globalna {d['wzrost_cm'].corr(d['log_market_value']):+.3f}, "
        f"DEF {d[d.pozycja == 'DEF']['wzrost_cm'].corr(d[d.pozycja == 'DEF']['log_market_value']):+.3f}, "
        f"FOR {d[d.pozycja == 'FOR']['wzrost_cm'].corr(d[d.pozycja == 'FOR']['log_market_value']):+.3f}"
    )

    # 6. Sklad osi "znani / nowi" z D-12 - liczby wchodza do protokolu w E4.
    for zbior in ["kalibracja", "test", "zbior_C"]:
        pod = model[model["podzial"] == zbior]
        zapisz_fakt(
            f"a12_udzial_znanych_{zbior}",
            round(100 * float(pod["znany_z_treningu"].mean()), 1),
        )
    # Zawodnicy wystepujacy i w Big 5, i w Primeira Lidze.
    w_big5 = set(big5(model)["player_id"])
    w_c = set(model.loc[model["podzial"] == "zbior_C", "player_id"])
    zapisz_fakt("a12_zawodnikow_w_obu_dziedzinach", len(w_big5 & w_c))

    fig, osie = plt.subplots(1, 3, figsize=(13, 3.6))

    dni = sezony["dni_od_30_06"]
    osie[0].hist(dni, bins=60, color="#4C72B0")
    osie[0].axvline(0, color="#C44E52", linewidth=1)
    osie[0].set(
        title="A12. Kiedy powstaje wycena",
        xlabel="dni od 30 czerwca",
        ylabel="liczba par",
    )

    osie[1].scatter(
        b["minuty_sezon"], b["log_market_value"], s=3, alpha=0.12, color="#4C72B0"
    )
    osie[1].set(
        title=f"A12. Wolumen gry a wartość, r = {korelacje['minuty_sezon']:.3f}",
        xlabel="minuty w sezonie",
        ylabel="log1p(wartość)",
    )

    for poz, kolor in zip(POZYCJE, ["#4C72B0", "#55A868", "#C44E52"]):
        s = d[d["pozycja"] == poz]
        kosze = pd.cut(s["wzrost_cm"], bins=[150, 172, 176, 180, 184, 188, 210])
        krzywa = s.groupby(kosze, observed=True)["log_market_value"].median()
        osie[2].plot(
            range(len(krzywa)), krzywa.to_numpy(), "o-", color=kolor, label=poz
        )
    osie[2].set_xticks(range(6))
    osie[2].set_xticklabels(
        ["<172", "172-6", "176-80", "180-4", "184-8", ">188"], fontsize=7
    )
    osie[2].set(title="A12. Wzrost a wartość wewnątrz pozycji", xlabel="wzrost [cm]")
    osie[2].legend(fontsize=8)
    zapisz_figure(fig, "a12_ciekawostki")
