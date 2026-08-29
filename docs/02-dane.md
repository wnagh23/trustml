# Dane

Wersja 1.1, 2026-08-27. Wszystkie liczby zmierzone na `data/processed/model.parquet`
z 2026-08-24.

## Źródła

### FBref — statystyki gry

`data/raw/fbref/master.db`, SQLite, 221 MB, zero indeksów. Klucz złożony
`(match_id, player_id)`. `player_id` to ośmioznakowy hash FBref, na przykład
`d5326924`, stabilny między sezonami.

| tabela | wiersze | zawartość |
|---|---|---|
| Match | 17 551 | sezon, data, drużyny, xG, wynik |
| Player\_Info | 521 261 | nazwa, narodowość, pozycja, wiek, minuty |
| Summary | 521 261 | gole, strzały, xG, npxG, SCA, GCA |
| Passing | 521 261 | podania krótkie, średnie, długie, xAG |
| Possession | 521 261 | dotknięcia strefowe, drybling, prowadzenia |
| Defensive\_Actions | 521 261 | odbiory, przechwyty, bloki, błędy |
| Pass\_Types | 521 261 | prostopadłe, zmiany strony, dośrodkowania |
| Miscellaneous | 521 261 | faule, spalone, pojedynki powietrzne |
| Goalkeeper | 35 510 | nieużywana, bramkarzy odrzucamy (D-07) |

Zmienna celu pochodzi z Transfermarktu.

### Transfermarkt — zmienna celu i cechy statyczne

`data/raw/tm/`, snapshot Kaggle `davidcariboo/player-scores`. Używamy dwóch plików:
`players.csv` (50 149 wierszy — data urodzenia, wzrost, noga) oraz
`player_valuations.csv` (507 815 wierszy — wyceny z datami). Pozostałe pliki
w tym katalogu zostają nietknięte.

Katalog `data/raw/` jest tylko do odczytu.

## Pipeline

Uruchomienie: `python case_study/football/przygotuj_dane.py`, około 20 minut.

Skrypt, bo notatnik zapisuje wyniki w pliku i każde uruchomienie
robi ogromny diff w gicie. Czytelna historia zmian ma bezpośrednie znaczenie dla W6.

| etap | moduł | co robi |
|---|---|---|
| 1 | `zrodla.py` | wczytanie tabel, deduplikacja, usunięcie redundancji |
| 2 | `zrodla.py` | połączenie sześciu tabel polowych po kluczu |
| 3 | `cechy.py` | parsowanie wieku, narodowości, pozycji |
| 4 | `cechy.py` | agregacja mecz → zawodnik × sezon |
| 5 | `cechy.py` | wskaźniki procentowe i normalizacja per 90 minut |
| 6 | `wycena.py` | crosswalk do Transfermarktu, cechy statyczne |
| 7 | `wycena.py` | dołączenie wyceny z okna kwiecień-wrzesień |
| 8 | `zbiory.py` | podział na trening, kalibrację, test i zbiór C |
| 9 | `zbiory.py` | deflacja, indeks liczony wyłącznie na treningu |
| 10 | `przygotuj_dane.py` | zapis plików |

Parametry — progi, ścieżki, listy sezonów, mapowania — siedzą w `parametry.py`.

Wyjście trafia do `data/processed/`: `zawodnik_sezon.parquet` (pełny zbiór kontrolny
z surowymi sumami sezonowymi), `model.parquet` (wąski zbiór modelowy) oraz
`manifest.json` (prowieniencja — sumy kontrolne, wersje i parametry przebiegu).

## Reguły, które trzeba znać

### Wskaźniki procentowe: suma liczników przez sumę mianowników

Uśrednianie procentów z poziomu meczu daje wynik fałszywy. Zawodnik z jednym
dryblingiem na dwa w meczu A (50 procent) i zerem na osiem w meczu B (0 procent) ma średnią procentów
25 procent, a poprawnie jeden na dziesięć, czyli 10 procent. Zmierzony błąd naiwnej
średniej sięga 33 punktów procentowych.

Agregacja zachowuje liczniki i mianowniki, a wskaźnik liczy się dopiero na poziomie
sezonu. Osiem wskaźników, definicje w `parametry.WSKAZNIKI_PROCENTOWE`; każda para
została zweryfikowana i odtwarza procent FBref w stu procentach wierszy.

Kolumna nazywa się `tackled_perecentage` — to literówka w schemacie źródłowym.
Zostawiamy ją w bazie i mapujemy przy odczycie.

### Normalizacja per 90 minut

Wszystkie statystyki licznikowe przeliczamy jako `suma_sezonowa / (minuty / 90)`.
Bez tego model uczy się głównie tego, ile zawodnik grał, co jest jednocześnie
skutkiem i przyczyną jego wartości.

`minuty_sezon` i `mecze` zostają jako osobne cechy — sam wolumen gry ma własną
wartość informacyjną, niezależną od produktywności.

### Pozycja i region

Pozycja jest zapisana per mecz: 1 119 unikalnych wartości, a 86,9 procent par
zawodnik-sezon ma więcej niż jedną. Reguła jest dwustopniowa. Z wartości złożonej
bierzemy pierwszy kod, bo kolejność ma znaczenie (`FW,AM` to FOR, `AM,FW` to MID).
Potem wybieramy klasę o największej sumie minut w sezonie.

Klasy: DEF to CB, LB, RB, WB; MID to DM, CM, LM, RM, AM; FOR to LW, RW, FW.

Region wyliczamy z kodu kraju na trzy wartości: `EUROPA` według UEFA, więc z Izraelem,
Turcją, Armenią, Azerbejdżanem, Gruzją, Kazachstanem i Rosją; `AMERYKA_PLD` według
CONMEBOL; oraz `RESZTA`. Surowa narodowość dałaby przy 126 wartościach kategorie
o liczebności jednostkowej.

### Wiek

`Player_Info.age` to tekst w formacie lata-dni, na przykład `'20-261'`. Z niego
i z daty meczu wyliczamy datę urodzenia, a z niej wiek na 30 czerwca roku kończącego
sezon. Wiek liczymy raz, na tę ustaloną datę — uśrednianie po meczach dałoby
zawodnikowi grającemu cały sezon i grającemu tylko wiosnę różne wieki mimo tej
samej daty urodzenia.

Cecha `wiek_do_kw`, czyli kwadrat wieku, wchodzi do zbioru, bo wartość rośnie mniej
więcej do dwudziestego piątego roku życia i potem spada. Drzewa wychwycą ten kształt
same, modelom liniowym trzeba go podać jawnie — dzięki tej cesze porównanie rodzin
modeli jest uczciwsze.

### Crosswalk FBref → Transfermarkt

Łączymy po znormalizowanej nazwie: NFKD, ASCII, małe litery, tylko litery i spacje.
Nazwy występujące w Transfermarkcie więcej niż raz odrzucamy w całości przez
`drop_duplicates(keep=False)` — lepiej stracić wiersz niż przypisać cudzą wycenę.
Dodatkowa kontrola: rozbieżność daty urodzenia powyżej siedmiu dni odrzuca
dopasowanie.

Pokrycie wynosi około 86 procent par. Strata ma charakter systematyczny — giną nazwiska
iberyjskie i brazylijskie. Odnotowane jako ograniczenie.

### Moment wyceny

Dla sezonu kończącego się w roku t bierzemy okno od 1 kwietnia do 30 września roku t
i wybieramy wycenę o dacie najbliższej 30 czerwca. Maksimum z okna byłoby estymatorem
obciążonym w górę, a wielkość obciążenia zależy od liczby obserwacji, która jest
skorelowana z popularnością zawodnika.

To jest model wyceny bieżącej. Tak też nazywamy go w całej pracy.

### Deflacja

Używamy łańcuchowego indeksu o stałym składzie wiekowym. Dla każdej pary kolejnych
sezonów liczymy średnią `log(wartość)` w komórkach wiek × pozycja (przedziały wieku
0, 20, 22, 24, 26, 28, 30, 32, 99, trzy pozycje, komórki poniżej dwudziestu
obserwacji pomijamy), bierzemy różnicę średnich wewnątrz komórki i uśredniamy ważąc
liczebnością sezonu wcześniejszego.

Porównujemy zawodników w tym samym wieku, dobieranych osobno w każdym sezonie —
inaczej starzenie się mieszałoby się z inflacją. Odrzucony wariant na stałym panelu
dawał wniosek, że realne wartości potroiły się w osiem lat, czyli wynik ewidentnie zawyżony.

Średnia, bo Transfermarkt wycenia w okrągłych progach (123 różne
wartości w całym zbiorze), więc mediana komórki o sześćdziesięciu do stu
czterdziestu obserwacjach skacze między progami i daje „inflację" rzędu 60 procent
tam, gdzie rynek ledwie drgnął.

Indeks liczymy wyłącznie na sezonach treningowych. Sezon kalibracyjny, testowy
i zbiór C dostają współczynnik ostatniego sezonu treningowego. Policzenie ich własnych
oznaczałoby użycie informacji ze zbioru testowego do przekształcenia celu — wyciek
subtelny, ukryty w zmiennej objaśnianej.

Zmierzona inflacja w Big 5 wynosi 38 procent między 2017-2018 a 2023-2024, a poziom
testu leży 18,4 procent powyżej treningu. Zapisane są oba cele; wybór głównego
opisuje D-04.

## Zbiór modelowy

`data/processed/model.parquet` — 13 709 wierszy na 114 kolumn, 5 310 zawodników.
Klucz `(player_id, season)` jest unikalny.

### Podział

| zbiór | wierszy | sezony | ligi |
|---|---|---|---|
| trening | 7 924 | 2017-18, 2018-19, 2020-21, 2021-22 | Big 5 |
| kalibracja | 1 982 | 2022-23 | Big 5 |
| test | 1 957 | 2023-24 | Big 5 |
| zbiór C | 1 846 | 2018-19 do 2023-24 | Primeira Liga |

Podział jest czasowy. Zbiór kalibracyjny istnieje wyłącznie dla W3;
kalibracja predykcji konforemnej na treningu daje zawyżone pokrycie.

Odrzucone sezony: 2019-2020, bo Ligue 1 rozegrała 279 z 380 meczów; 2024-2025,
bo ma 25 procent braków celu skorelowanych z ligą (La Liga 54,8 procent pokrycia
wobec Bundesligi 89,3); oraz 2025-2026, bo sezon trwa. Po odrzuceniu pokrycie wycen
w pozostałych sezonach wynosi od 99,1 do 99,6 procent.

Zbiór C zaczyna się od sezonu 2018-2019 — Primeira Liga była scrapowana rok później
niż Big 5, więc granica pochodzi ze źródła (O-2 w `04-plan.md`).

### Kolumny

Klucz i metadane, pięć kolumn: `player_id`, `season`, `name`, `tm_player_id`,
`podzial`, plus `znany_z_treningu` (D-12).

Cele, cztery kolumny: `market_value_in_eur`, `market_value_real`, `log_market_value`,
`log_market_value_real`.

Cechy: 87 statystyk `_p90`, 8 wskaźników procentowych, 5 liczbowych (`wiek`,
`wiek_do_kw`, `wzrost_cm`, `minuty_sezon`, `mecze`) i 5 kategorycznych (`pozycja`,
`region`, `liga`, `noga`, `zmienil_lige`). Razem 105 cech.

Kategoryczne zostają tekstem — kodowanie ma się dziać wewnątrz `Pipeline`, żeby
zestaw kategorii był ten sam we wszystkich zbiorach (D-10).

### Rozkłady

Cel ma medianę 5,0 mln euro przy zakresie od 25 tysięcy do 200 milionów. Skośność
nominalna wynosi 3,62, po `log1p` spada do −0,13, czyli rozkład jest praktycznie
symetryczny.

Minuty: próg 225, mediana 1 534, maksimum 3 420. Wiek od 16 do 43, mediana 26.
Pozycje: DEF 5 531, MID 4 684, FOR 3 494. Regiony: EUROPA 9 651, RESZTA 2 218,
AMERYKA\_PLD 1 840. Noga: prawa 9 707, lewa 3 619, obie 372, brak 11. Ligę w trakcie
sezonu zmieniło 2,75 procent zawodników.

Mediana wartości według ligi, w milionach euro: Premier League 15,0; Bundesliga 5,0;
Serie A 5,0; La Liga 4,0; Ligue 1 4,0; Primeira Liga 0,8.

### Braki

Braki są wyłącznie strukturalne i sięgają maksymalnie 2,2 procent:
`dribble_success_percentage` 295, `tackled_perecentage` 295,
`successful_dribbler_tackle_percentage` 40, `aerials_won_percentage` 13,
`noga` 11, `long_completion_percentage` 9, `wzrost_cm` 2.

Zostają jako NaN. To jest informacja — zawodnik miał przez cały sezon zero prób danej
akcji — czyli treść sama w sobie. Imputacja odbywa się wewnątrz `Pipeline` (D-09).

## Ograniczenia

Czym jest zmienna celu. Wycena Transfermarkt jest agregatem opinii społeczności
moderowanym przez redakcję. Cena transakcyjna to inna wielkość — zależy od długości
kontraktu, sytuacji finansowej klubu, klauzul odstępnego i konkurencji o zawodnika,
a te czynniki pozostają poza danymi. Model odtwarza konsensus opinii o zawodniku. Jeśli wyceny są obciążone, model
odziedziczy to obciążenie — bezpośrednie połączenie z W5.

Crosswalk gubi około 14 procent par, i to systematycznie: giną nazwiska iberyjskie
i brazylijskie. Świadomie zaakceptowane, bo zbiór jest przykładem demonstracyjnym.

Bramkarze są poza zakresem, czyli około 10 procent populacji zawodników.

Popularność medialna została porzucona. Próby pozyskania danych z Wikipedia pageviews
i Google Trends rozbiły się o problem łączenia encji, niekompletne pokrycie
historyczne i limity API. Część niewyjaśnionej wariancji pochodzi prawdopodobnie
z rozgłosu, który pozostaje poza zbiorem cech. Jest to przykład obciążenia zmienną pominiętą —
i hipoteza o mechanizmie obciążenia w W5, jeśli rozgłos jest nierówno rozłożony
geograficznie.

Próg 225 minut usuwa około 19 procent par, systematycznie: wypadają młodzi, rezerwowi
i kontuzjowani, czyli systematycznie tańsi. Próg jest parametrem, więc analiza
wrażliwości dla wartości 0, 225, 450 i 900 kosztuje jedno przełączenie.

Zawodnicy znani modelowi. Podział jest czasowy, więc ten sam zawodnik występuje
w wielu zbiorach w różnych sezonach. Pary są rozłączne, więc formalnie podział jest
czysty, ale model rozpoznaje zawodnika po kombinacji wzrostu, wieku, pozycji i profilu
statystycznego, mimo że `player_id` zostaje poza cechami. Zmierzone modelem
HistGradientBoosting: na teście znani mają R² 0,68 wobec 0,56 dla nowych, a w zbiorze
C znani 0,51 wobec −0,50 dla nowych. Przewaga zostaje po wyrównaniu wieku grup,
więc bierze się z samej znajomości zawodnika. Udział znanych to 66 procent na teście (1 295
z 1 957) i 9 procent w zbiorze C. Konsekwencje raportowania opisuje D-12.

Trzysta czterdziestu siedmiu zawodników występuje i w Big 5, i w Primeira Lidze,
w różnych sezonach. Jest to dopuszczalne, a zarazem wzmacnia potrzebę stratyfikacji
z D-12.

Dane osobowe. Zbiór zawiera nazwiska, wiek, narodowość i wzrost realnych osób. Dane
są publiczne i dotyczą działalności zawodowej, ale fakt należy odnotować w pracy.
Przetworzone dane leżą poza repozytorium.

## Dług

Prowieniencja danych surowych jest zerowa. `master.db` przychodzi bez metadanych:
`user_version` wynosi 0, `application_id` wynosi 0, brak indeksów i brak tabeli
z wersją, datą scrape'u czy źródłowym adresem. Jedyne daty pośrednie to ostatni mecz
FBref 2026-01-15 i ostatnia wycena Transfermarkt 2026-02-27, czyli dwa źródła
pochodzą z momentów oddalonych o pół roku.

To największy dług wymiaru W6. Spłata: `manifest.json` w etapie 10 z sumami SHA-256
każdego pliku wejściowego, wersjami pandas, numpy i Pythona, datą uruchomienia,
wszystkimi parametrami i liczbami wierszy po każdym etapie. Zrobione
w `case_study/football/manifest.py`.

Archiwa ZIP zaudytowane (`scripts/audyt_zrodel.py`, wynik w `docs/zrodla_fbref.json`).
Zawierają 17 551 surowych stron FBref nazwanych identyfikatorem meczu, każda z adresem
kanonicznym `https://fbref.com/en/matches/<match_id>/...` w nagłówku i datą pobrania
w metadanych archiwum. Liczba stron zgadza się co do jednego z liczbą meczów w bazie,
w obie strony — `master.db` powstał dokładnie z tych scrape'ów.

Okna pobrania: Big 5 od 2023-02-14 do 2026-01-17, Primeira Liga od 2024-11-08 do
2026-01-12. Prowieniencja `master.db` jest więc odtworzona, mimo że sama baza nadal
przychodzi bez metadanych.

Licencje do sprawdzenia przed złożeniem pracy: warunki użytkowania FBref wobec
scrapingu i wymagana forma cytowania, oraz licencja zbioru Kaggle
`davidcariboo/player-scores`.

## Powiązane dokumenty

- `01-projekt.md` — cel pracy, wymiary W1–W6, modele
- `03-decyzje.md` — log decyzji projektowych
- `04-plan.md` — mapa drogowa i status
- `05-eda.md` — analiza eksploracyjna, wnioski i ciekawostki
