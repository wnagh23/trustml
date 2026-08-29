# Plan i status

Wersja 1.1, 2026-08-27

## Mapa drogowa

| etap | zakres | status |
|---|---|---|
| E1 | pipeline danych | `[x]` |
| E2 | porządki: manifest, testy, importy | `[x]` |
| E3 | analiza eksploracyjna | `[x]` |
| E4 | protokół walidacji i poziom odniesienia | `[ ]` |
| E5 | modele M1–M3, wymiar W1 | `[ ]` |
| E6 | trustml 0.1: szkielet, W1, W6 | `[ ]` |
| E7 | odporność, wymiar W2 | `[ ]` |
| E8 | niepewność, wymiar W3 | `[ ]` |
| E9 | wyjaśnialność, wymiar W4 | `[ ]` |
| E10 | sprawiedliwość, wymiar W5 | `[ ]` |
| E11 | przesunięcie dziedziny: Primeira Liga | `[ ]` |
| E12 | trustml 1.0 i synteza | `[ ]` |

## E1 — pipeline danych

Status: zrobione.

Zbiór modelowy 13 709 na 114, 5 310 zawodników. Szczegóły w `02-dane.md`.

Artefakty: moduły `zrodla.py`, `cechy.py`, `wycena.py`, `zbiory.py`, `parametry.py`
i `przygotuj_dane.py` w `case_study/football/`, oraz `model.parquet`
i `zawodnik_sezon.parquet` w `data/processed/`.

Narzędzia: sqlite3, pandas 3.0, numpy, pyarrow, uv, ruff, Python 3.12.

### Wnioski

Parametr `validate=` w merge jest darmowy i wykrywa klasę błędów, które inaczej
ujawniają się dopiero jako niewytłumaczalne wyniki modelu. Poprzednia wersja
pipeline'u łączyła po nazwisku bez walidacji i miała 846 zduplikowanych par
zawodnik-sezon.

Naiwne uśrednianie procentów z poziomu meczu daje błąd do 33 punktów procentowych.
Wskaźnik musi być liczony jako suma liczników przez sumę mianowników na poziomie
sezonu.

Deflacja na stałym panelu, czyli na tych samych zawodnikach w kolejnych sezonach,
jest błędna: mierzy sumę inflacji rynku i starzenia się zawodników, a ponieważ
w próbie dominują zawodnicy po szczycie kariery, efekt starzenia przeważa. Dawała
wniosek, że realne wartości potroiły się w osiem lat.

Mediana jest złym estymatorem środka komórki, gdy źródło raportuje wartości
w okrągłych progach — skacze między progami i daje „inflację" rzędu 60 procent tam,
gdzie rynek ledwie drgnął. Użyliśmy średniej z logarytmu wartości.

Braki w kolumnach procentowych są strukturalne: mianownik równy zeru.
Po agregacji sezonowej spadają z około 50 procent do około 2 procent i wtedy są
informacją samą w sobie.

Filtr minimalnych minut jest konieczny, ale usuwa dane systematycznie: wypadają młodzi,
rezerwowi i kontuzjowani, czyli systematycznie tańsi.

## E2 — porządki

Status: zrobione.


- `[x]` `manifest.json` w etapie 10: SHA-256 każdego pliku wejściowego, wersje pandas,
  numpy i Pythona, data uruchomienia, wszystkie parametry z `parametry.py`, liczby
  wierszy po każdym etapie. To pierwszy realny kawałek W6 i kosztuje pół godziny.
- `[x]` kolumna `znany_z_treningu` w etapie 8 (D-12). Bez niej każda stratyfikacja
  wymagałaby przeliczania w każdym notatniku osobno.
- `[x]` `tests/test_dane.py` — przenieść asercje ze skryptu do pytest. Dziś sprawdzenia
  odpalają się tylko przy pełnym przebiegu pipeline'u, czyli po dwudziestu minutach.
  Test czytający `model.parquet` biegnie w sekundę i wykryje, że plik został
  podmieniony. Zakres: unikalność klucza, brak Primeira Liga w treningu i kalibracji,
  minuty powyżej progu, wskaźniki w zakresie od zera do stu albo NaN, rozłączność par
  między zbiorami, oczekiwane kształty i liczebności.
- `[x]` naprawić importy. Moduły robią `from parametry import ...`, więc działają tylko
  z konkretnego katalogu roboczego — dokładnie ta klasa błędów, przed którą miał
  chronić layout `src` (D-19).
- `[x]` `README.md` — jak odtworzyć zbiór, jak odpalić testy.
- `[x]` pre-commit: `ruff format`, `ruff check`, `nbstripout`.

### Wnioski

Prowieniencja `master.db` była odzyskiwalna, mimo że baza przychodzi bez metadanych. Archiwa
ZIP obok niej zawierają 17 551 surowych stron FBref z adresem kanonicznym w nagłówku
i datą pobrania w metadanych pliku. Liczba stron zgadza się co do jednej z liczbą
meczów w bazie, w obie strony. Audyt: `scripts/audyt_zrodel.py`, wynik
w `docs/zrodla_fbref.json`, streszczenie w manifeście.

Przy okazji rozstrzygnięte O-2: Primeira Liga zaczyna się w źródle od sezonu 2018-2019,
bo została doscrapowana rok później niż Big 5 (od 2024-11-08).

Testy czytające gotowy `model.parquet` biegną 1,1 s wobec 40 s pełnego przebiegu
pipeline'u. Siedemnaście sprawdzeń, w tym rozłączność par między zbiorami i zgodność
flagi `znany_z_treningu` z faktycznym składem.


## E3 — analiza eksploracyjna

Status: zrobione.

Celem było odpowiedzenie na pytania, których odpowiedzi zmienią decyzje w E4 i E5.

- `[x]` rozkład celu: nominalny, logarytmiczny, zdeflowany. Weryfikacja symetrii
  po `log1p`.
- `[x]` skuteczność deflacji: czy po zdeflowaniu mediana realna jest płaska w czasie.
  Odchylenie od płaskiej wskazuje błąd w liczeniu indeksu.
- `[x]` krzywa wiek-wartość. Weryfikacja kształtu odwróconego U, czyli uzasadnienia
  dla cechy `wiek_do_kw`. Kształt inny niż paraboliczny czyni tę cechę ozdobnikiem
  i trzeba to napisać.
- `[x]` struktura panelu: ilu zawodników w ilu sezonach, jak długi jest ogon
  jednosezonowy. Wejście do GroupKFold w E4.
- `[x]` rozkład celu i cech w podgrupach: pozycja, liga, region, wiek. Wstępna diagnoza
  W5 przed jakimikolwiek resztami modelu.
- `[x]` korelacje między cechami `_p90`: ile z 87 jest redundantnych przy współczynniku
  powyżej 0,95. Wejście do filtra korelacyjnego w Pipeline.
- `[x]` mapa braków: czy 295 braków w `dribble_success_percentage` rozkłada się losowo,
  czy koncentruje w jednej pozycji lub lidze. Jeśli koncentruje, imputacja medianą
  wprowadzi obciążenie grupowe, czyli problem dla W5.
- `[~]` wrażliwość na próg minut: 0, 225, 450 i 900. Zrobione dla 225, 450, 900
  i 1350. Wariant „próg 0" wymaga przebiegu pipeline z innym `MIN_MINUT`, bo zbiór
  modelowy jest już odfiltrowany. Do domknięcia przy następnym przebiegu danych.
- `[x]` sufit informacyjny: ile wariancji celu tłumaczy sam wiek plus pozycja plus liga,
  bez statystyk gry. Podaje realistyczne oczekiwanie na R².
- `[x]` PSI i test Kołmogorowa-Smirnowa dla cech Big 5 wobec zbioru C. Tylko cechy,
  bez zmiennej celu (D-14).

Dołożone poza planem, bo wyszło z pomiaru dryfu:

- `[x]` analiza nieciągłości w źródle: rozdzielenie trwałej zmiany poziomu cechy
  od anomalii jednosezonowej, z testem tożsamości księgowej wskaźników dryblingu.
- `[x]` pomiar kosztu filtra korelacyjnego na jakości predykcji, w dwóch wariantach
  reguły wyboru cechy z pary.

Narzędzia: matplotlib, scipy, scikit-learn. Seaborn i statsmodels okazały się zbędne.

Artefakty: `case_study/football/eda.py` i `uruchom_eda.py`, raport `docs/05-eda.md`,
czternaście figur w `reports/figures/eda/`, czternaście tabel w `reports/tables/eda/`
oraz `reports/tables/eda/fakty.json` ze 175 zmierzonymi liczbami. Notatnika nie ma —
logika mieszka w module, a raport cytuje wyłącznie liczby z `fakty.json`, więc każde
zdanie da się sprawdzić bez uruchamiania czegokolwiek.

### Wnioski

Kolumna `dribble_success_percentage` zmienia znaczenie dokładnie na granicy zbioru
treningowego. Do sezonu 2021-2022 skuteczność dryblingu i wskaźnik odbioru sumują
się do stu w 98,7 procent wierszy, bo próba dryblingu ma dwa możliwe zakończenia.
Od sezonu 2022-2023 tożsamość obowiązuje w 21,1 procent wierszy. Model uczy się tej
cechy w jednej definicji i jest oceniany w drugiej, a granica pokrywa się co do
sezonu z granicą podziału. Jest to najpoważniejsze ustalenie etapu i przedmiot O-6.

Wolumen gry, czyli minuty i mecze, podnosi R² o 0,177 ponad blok kontekstowy,
a wszystkie 95 statystyk boiskowych dokłada ponad to 0,092. Informacja „ile grał"
jest dla wyceny cenniejsza niż komplet informacji „jak grał". Ranking ważności w W4
prawie na pewno postawi `minuty_sezon` na czele, a interpretacja przyczynowa tej
cechy jest wykluczona, bo minuty są skutkiem wartości zawodnika w tym samym stopniu,
w jakim są jej przyczyną.

Zestaw 95 statystyk boiskowych jest pomiarem mniej więcej szesnastu rzeczy:
tyle składowych głównych zbiera 80 procent wariancji cech, przy wymiarowości
partycypacyjnej 8,0 i pierwszej składowej tłumaczącej 28,5 procent. Cechy układają
się w 28 bloków, z których największy liczy 21 kolumn opisujących objętość gry przy
piłce. Deklarowana liczba 105 cech przecenia bogactwo opisu zawodnika, więc warto
podawać obok niej liczbę efektywną.

Filtr korelacyjny przy progu 0,95 kosztuje 0,15 R² na modelu liniowym i reguła
wyboru cechy z pary tego nie zmienia. Regularyzacja grzbietowa radzi sobie ze
współliniowością sama. Uzasadnieniem dla filtra zostaje stabilność wyjaśnień w W4,
a nie dokładność w W1 — z ceną zmierzoną liczbowo.

Krzywa wieku nie ma kształtu odwróconego U: najdroższym rocznikiem są
osiemnastolatkowie z medianą 10,0 miliona euro. Efekt pochodzi z progu 225 minut,
który z tego rocznika przepuszcza wyłącznie zawodników z realną grą w Big 5. Sama
parabola broni się mimo to liczbowo — zbiera 96,6 procent informacji o wieku wobec
sufitu rocznikowego — więc `wiek_do_kw` zostaje.

Sufit informacyjny: sam kontekst, czyli sześć zmiennych niewymagających ani jednego
meczu, daje R² 0,313 na kalibracji. Pełny zestaw cech na HistGradientBoosting bez
strojenia daje 0,724. W tej skali należy czytać wyniki E5.

Braki są skoncentrowane pozycyjnie: 4,88 procent u obrońców wobec 0,17 u napastników
w obu wskaźnikach dryblingu. Imputacja medianą wprowadziłaby obciążenie grupowe,
więc dane przemawiają za wariantem z flagą „zero prób" z O-3.

Deflacja działa tam, gdzie liczono indeks: rozstęp poziomów na treningu spada
z 0,263 do 0,047. Poza treningiem zostaje 0,158 z konstrukcji płaskiego
przedłużenia, co jest znane co do mechanizmu.

Przesunięcie cech do zbioru C jest umiarkowane — dwie cechy z PSI powyżej 0,25,
mediana 0,033. Wobec przesunięcia poziomu celu o około 1,7 jednostki logarytmicznej
oznacza to, że degradacja w E11 będzie zdominowana przez składnik obciążenia,
co czyni rozkład z D-13 warunkiem sensowności głównego wyniku pracy.

## E4 — protokół walidacji i poziom odniesienia

Status: nierozpoczęte. Szacunek: dwa do trzech dni. Najważniejszy metodologicznie.

Celem jest ustalenie zasad oceny raz i trzymanie się ich do końca. Zmiana protokołu
po zobaczeniu wyników to najcichszy sposób na oszukanie samego siebie.

- `[ ]` `configs/split.yaml` — podział zamrożony, z hashem `model.parquet`.
- `[ ]` `src/trustml/evaluation/splits.py` — GroupKFold po `player_id` wewnątrz zbioru
  treningowego, do strojenia hiperparametrów.
- `[ ]` `tests/test_no_leakage.py` — przecięcie par `(player_id, season)` między
  zbiorami musi być puste. Osobno raport przecięcia samych `player_id`, jako świadoma
  i udokumentowana cecha podziału (D-12).
- `[ ]` `src/trustml/evaluation/metrics.py` — RMSLE, MdAPE, MAE, R², wszystkie
  z obowiązkowym parametrem stratyfikacji (D-12).
- `[ ]` transformacja odwrotna do euro: korekta Duana albo jawna deklaracja,
  że raportujemy medianę warunkową. Rozstrzygnąć raz.
- `[ ]` poziom odniesienia: M0a mediana globalna, M0b mediana w komórce pozycja × liga
  × sezon, M0c przepisanie wyceny z t−1 (D-06). M0c raportowany w dwóch wariantach:
  na podzbiorze z dostępnym t−1, czyli 71,6 procent testu, oraz na pełnym teście
  z uzupełnieniem M0b.

Kluczowe pytanie tego etapu: jak wysoko leży M0b i M0c. Ta liczba określa wymowę całej
pracy. W poprzedniej wersji zbioru M0c dawało R² 0,69 i bez tego punktu odniesienia
R² 0,88 modelu byłoby łatwo przecenić.

Artefakty: `configs/split.yaml`, `src/trustml/evaluation/metrics.py` i `splits.py`,
`tests/test_no_leakage.py`, `reports/tables/e4_baseline.csv`.

## E5 — modele M1–M3, wymiar W1

Status: nierozpoczęte. Szacunek: cztery do sześciu dni.

- `[ ]` wspólny Pipeline: `SimpleImputer`, filtr korelacyjny,
  `OneHotEncoder(handle_unknown="ignore")` bez `drop`, `StandardScaler`.
- `[ ]` M1 Elastic Net na `log1p(y)`, alfa i `l1_ratio` z walidacji krzyżowej.
- `[ ]` M2 Random Forest.
- `[ ]` M3 XGBoost, early stopping na zbiorze kalibracyjnym.
- `[ ]` strojenie przez optunę, około stu prób na model, wyłącznie GroupKFold
  na treningu.
- `[ ]` MLflow: każdy przebieg z hashem danych i configu.
- `[ ]` analiza reszt: heteroskedastyczność, regresja do średniej w ogonach, odchylenia
  per liga i pozycja.
- `[ ]` porównanie celu nominalnego i zdeflowanego (D-04). Hipoteza: zysk pojawi się
  u modeli liniowych, drzewa zostaną na swoim poziomie.
- `[ ]` test jednostkowy na pułapkę `drop="first"` z D-10.

Artefakty: `src/trustml/models/`, `models/*.joblib`, `mlruns/`,
`reports/tables/e5_w1.csv`, `notebooks/02-modelowanie.ipynb`.

## E6 — trustml 0.1

Status: nierozpoczęte. Szacunek: cztery do sześciu dni.

Celem jest postawienie pakietu, zanim eksperymentów będzie za dużo, żeby dało się je
ujednolicić.

- `[ ]` klasa `TrustReport` z API opisanym w `01-projekt.md`.
- `[ ]` `evaluation/correctness.py` — wymiar W1.
- `[ ]` `evaluation/reproducibility.py` — wymiar W6: wariancja ziarnowa, hash danych,
  determinizm.
- `[ ]` renderer raportu HTML na jinja2.
- `[ ]` testy jednostkowe, pokrycie powyżej 70 procent.
- `[ ]` CI na GitHub Actions: ruff i pytest przy każdym pushu.
- `[ ]` rozstrzygnąć, czym `trustml` różni się od deepchecks, giskard, evidently
  i fairlearn. Przez lekturę ich dokumentacji. Bez odpowiedzi
  wkład pracy jest podważalny na obronie.

## E7 — odporność, wymiar W2

- `[ ]` rolling origin: trenuj na sezonach do t, testuj na t+1, przesuwaj t. Wynikiem
  jest krzywa degradacji w czasie. Uwaga na lukę kalendarzową
  między 2018-2019 a 2020-2021 (D-08).
- `[ ]` szum gaussowski o sigma 1, 5 i 10 procent na cechach `_p90`.
- `[ ]` maskowanie 5 i 10 procent losowych cech z imputacją z treningu.
- `[ ]` perturbacja skierowana: zaburzenie wyłącznie najważniejszych cech według SHAP
  i porównanie z zaburzeniem losowym.
- `[ ]` metryka zbiorcza odporności jako nachylenie krzywej degradacji.
- `[ ]` `evaluation/robustness.py`.

## E8 — niepewność, wymiar W3

- `[ ]` predykcja konforemna przez MAPIE, alfa 0,1 i 0,2, kalibracja na sezonie
  2022-2023.
- `[ ]` pokrycie empiryczne wobec nominalnego oraz średnia szerokość przedziału —
  zawsze razem.
- `[ ]` pokrycie warunkowe według pozycji, ligi, decyla wartości, wieku oraz osi
  znani/nowi (D-12).
- `[ ]` porównanie z alternatywami: regresja kwantylowa w XGBoost, rozrzut drzew
  w Random Forest.
- `[ ]` `evaluation/uncertainty.py`.

Uwaga metodologiczna: predykcja konforemna zakłada wymienialność obserwacji, którą
podział temporalny łamie. Trzeba to opisać jawnie i pokazać liczbowo, jak duży jest
efekt.

## E9 — wyjaśnialność, wymiar W4

- `[ ]` SHAP: `TreeExplainer` dla drzew, `LinearExplainer` dla Elastic Net.
- `[ ]` permutation importance jako niezależna weryfikacja.
- `[ ]` PDP i ALE dla wieku, minut i xG na 90 minut.
- `[ ]` stabilność: dziesięć przebiegów z różnymi ziarnami, ranking cech według
  średniego modułu SHAP, rozkład tau Kendalla dla wszystkich par rankingów, pokrycie
  pierwszej dziesiątki między przebiegami.
- `[ ]` sanity check dziedzinowy: czy najważniejsze cechy są sensowne merytorycznie.
- `[ ]` `evaluation/explainability.py`.

## E10 — sprawiedliwość, wymiar W5

- `[ ]` grupy: pozycja, liga, region, przedział wieku, decyl wartości.
- `[ ]` parytet jakości (różnica RMSLE i MdAPE) oraz parytet obciążenia (średnia
  reszty) — dwa niezależne pomiary.
- `[ ]` coverage gap, czyli różnica pokrycia konforemnego między grupami. Spina W3 z W5.
- `[ ]` bootstrap na istotność różnic międzygrupowych, obok wartości punktowych.
- `[ ]` analiza źródła obciążenia: czy luka pochodzi z modelu, czy jest odziedziczona
  po wycenach Transfermarkt.
- `[ ]` `evaluation/fairness.py`.

Język opisu: mierzymy obciążenie wycen. Wnioski formułujemy o rynku i jego wycenach.

## E11 — przesunięcie dziedziny: Primeira Liga

Jeden eksperyment, w którym wszystkie wymiary są mierzone naraz.

- `[ ]` zmierzyć wielkość przesunięcia cech: PSI i test Kołmogorowa-Smirnowa. Można
  wcześniej (D-14).
- `[ ]` pełny `TrustReport` dla wszystkich modeli, w wariantach z ligą i bez (D-11).
- `[ ]` rozkład degradacji na obciążenie kierunkowe i rozrzut (D-13). Obowiązkowo —
  od tego zależy wymowa wyniku.
- `[ ]` porównanie główne na podzbiorach „nowi vs nowi" (D-12); wersja „całość vs
  całość" raportowana obok.
- `[ ]` wariant z rekalibracją przesunięcia na małej próbce ze zbioru C.
- `[ ]` W3: załamanie pokrycia konforemnego — hipoteza główna eksperymentu.
- `[ ]` W4: czy ranking SHAP zmienia się między dziedzinami.
- `[ ]` analiza: które wymiary degradują pierwsze i najmocniej.

Rozszerzenie: drabinka lig, od 0,8 mln euro w Primeira Lidze po 15 mln w Premier
League, pozwala zbudować krzywą dawka-odpowiedź o kilku punktach pomiarowych.

## E12 — trustml 1.0 i synteza

- `[ ]` wszystkie moduły W1–W6 zintegrowane w `TrustReport`.
- `[ ]` `scripts/run_all.py` — jedna komenda odtwarzająca wszystkie wyniki.
- `[ ]` test przejścia na ACSIncome z folktables, bez zmiany linii w `src/`.
- `[ ]` pokrycie testami powyżej 80 procent, zielone CI.
- `[ ]` macierz zbiorcza: modele w wierszach, W1–W6 w kolumnach, z rankingiem w każdej
  kolumnie.
- `[ ]` tau Kendalla między rankingami według różnych wymiarów. To jest liczba
  potwierdzająca lub obalająca założenie robocze pracy.
- `[ ]` test istotności: czy różnice między modelami przekraczają wariancję ziarnową.
- `[ ]` dyskusja, dlaczego agregowanie wymiarów o różnych jednostkach w jeden indeks
  jest problematyczne.
- `[ ]` wersja 1.0.0, tag w gicie, `CITATION.cff`.

## Pytania otwarte

O-1. Czy jeden sezon testowy, czyli 1 957 par, wystarczy jako główny wynik, czy przejść
na rolling origin już w E4. Koszt: mniej danych treningowych w każdym oknie. Zysk:
krzywa zamiast punktu i odporniejszy wynik główny. Termin: przed E4.

O-2. ~~Zbiór C zaczyna się dopiero od sezonu 2018-2019.~~ Rozstrzygnięte 2026-08-27
audytem archiwów: granica pochodzi ze źródła. Primeira Liga była scrapowana rok
później niż Big 5 (od 2024-11-08) i tylko od sezonu 2018-2019. Konsekwencja dla E11:
zbiór C obejmuje o rok krótszą historię niż trening.

O-3. Czy wskaźniki procentowe z brakiem strukturalnym powinny dostać flagę
„zero prób" obok imputacji, czy samą imputację (D-09). Termin: przy budowie Pipeline
w E5.

O-4. Czym `trustml` różni się od deepchecks, giskard, evidently i fairlearn.
Termin: najpóźniej E6.

O-5. Wersjonowanie danych: DVC, Git LFS, czy zostawić poza repozytorium. Dziś 2,4 GB
leży poza gitem. Termin: przed złożeniem.

O-6. Co zrobić z cechami, które zmieniły definicję w źródle. Trzynaście cech ma trwały
krok poziomu powyżej 0,2 odchylenia standardowego, a w przypadku pary wskaźników
dryblingu granica zmiany pokrywa się co do sezonu z granicą między treningiem
a kalibracją (A11 w `05-eda.md`). Warianty: usunąć te cechy ze zbioru, zostawić
z adnotacją w ograniczeniach, albo wyrównać poziom per sezon — przy czym wyrównanie
liczone na wszystkich sezonach byłoby wyciekiem, więc musiałoby korzystać wyłącznie
z treningu i dzielić los deflacji z etapu 9. Termin: przed treningiem w E5.

O-7. Czy filtr korelacyjny w ogóle wchodzi do Pipeline. Zmierzony koszt przy progu
0,95 to 0,15 R² na modelu liniowym, a zysk leży w stabilności rankingu SHAP, czyli
w W4. Rozstrzygnięcie wymaga zmierzenia obu stron na tej samej konfiguracji: tau
Kendalla między rankingami z filtrem i bez. Termin: E5 razem z W4, ponieważ wcześniej
brakuje drugiej połowy bilansu.

## Powiązane dokumenty

- `01-projekt.md` — cel pracy, wymiary W1–W6, modele
- `02-dane.md` — źródła, pipeline, słownik zbioru, ograniczenia
- `03-decyzje.md` — log decyzji projektowych
- `05-eda.md` — analiza eksploracyjna, wnioski i ciekawostki
