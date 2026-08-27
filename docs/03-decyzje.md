# Dziennik decyzji

Wersja 1.1, 2026-08-27

Format uproszczonego ADR: kontekst, decyzja, uzasadnienie, konsekwencje. Wpisów nie
usuwamy. Jeśli decyzja się zmienia, dodajemy nowy wpis z adnotacją, który zastępuje.

Log zaczyna się od nowa wraz z przepisaniem pipeline'u danych w sierpniu 2026.
Poprzedni dziennik dotyczył innego pipeline'u i został wycofany; decyzje, które nadal
obowiązują, są tu przepisane pod nowymi numerami.

Statusy: przyjęta, otwarta, warunkowa, zastąpiona.

## D-01 — framework ewaluacji, nie porównanie modeli

Status: przyjęta

Naturalną ścieżką dla tego zadania jest porównanie kilku algorytmów i wybór
najdokładniejszego. Taki wynik jest jednak mało użyteczny i łatwy do uzyskania.

Obiektem pracy jest procedura ewaluacji wiarygodności; modele są jej przypadkami
testowymi, a dane piłkarskie przykładem demonstracyjnym.

Konsekwencja: głównym wynikiem jest macierz model × wymiar i analiza rozbieżności
rankingów przez tau Kendalla, nie tabela z R². Więcej pracy idzie w ewaluację niż
w modelowanie i taki jest zamierzony rozkład wysiłku.

## D-02 — sześć wymiarów wiarygodności

Status: przyjęta

W1 poprawność, W2 odporność, W3 niepewność, W4 wyjaśnialność, W5 sprawiedliwość,
W6 odtwarzalność. Operacjonalizacja w `01-projekt.md`.

Wymiary pokrywają mierzalną ilościowo część wymogów HLEG i NIST AI RMF, a każdy da
się zoperacjonalizować na zadaniu regresyjnym. Pominęliśmy wymogi HLEG dotyczące
prywatności, bezpieczeństwa i nadzoru człowieka — nie są mierzalne w tym zadaniu
i rozmyłyby zakres pracy.

Konsekwencja porządkowa: przy każdym użyciu oznaczenia W1–W6 dopisujemy, jaka metryka
za nim stoi. Obowiązuje w rozmowie, kodzie i dokumentacji.

## D-03 — zmienna celu: log1p(market\_value\_in\_eur)

Status: przyjęta

Skośność rozkładu nominalnego wynosi 3,62, a po `log1p` spada do −0,13, czyli rozkład
staje się praktycznie symetryczny.

Konsekwencja: metryki liczymy na skali logarytmicznej, przede wszystkim RMSLE. Przy
raportowaniu w euro konieczna jest korekta Duana albo jawna deklaracja, że podajemy
medianę warunkową — `expm1` ze średniej z logarytmów nie jest średnią.

## D-04 — cel główny nominalny, zdeflowany jako analiza wrażliwości

Status: przyjęta, warunkowa

Inflacja wycen w Big 5 wynosi 38 procent między 2017-2018 a 2023-2024, a poziom testu
leży 18,4 procent powyżej treningu. Przy podziale czasowym model powinien więc
systematycznie zaniżać na teście.

Pomiar tego nie potwierdza. HistGradientBoosting na teście dał MAE 0,5757 z deflacją
wobec 0,5734 bez niej. Drzewa nie ekstrapolują poza zakres treningu, więc różnica
poziomów przekłada się na obciążenie słabiej, niż sugeruje porównanie średnich.

Celem głównym jest `log_market_value` w wersji nominalnej. `market_value_real`
i `log_market_value_real` zostają w zbiorze jako wariant do analizy wrażliwości.

Warunek utrzymania: pomiar dotyczył jednego modelu drzewiastego. Modele liniowe
ekstrapolują i mogą zachować się inaczej — to jest właściwa treść eksperymentu
porównującego oba cele. Jeśli Elastic Net wyraźnie zyska na deflacji, decyzja wraca
do rozpatrzenia.

## D-05 — usunięcie market\_value\_prev ze zbioru cech

Status: przyjęta

Zeszłoroczna wycena wyjaśnia 83,75 procent wariancji celu w skali logarytmicznej.
Cecha nie wchodzi do zbioru modelowego.

Powody są dwa. Po pierwsze, zdominowałaby ranking SHAP: byłby jeden pewny zwycięzca
i szum poniżej, więc tau Kendalla wyszłoby wysokie i nie mierzyłoby niczego, a W4
stałby się bezprzedmiotowy. Po drugie, cecha nie istnieje dla debiutantów, czyli
około 28 procent zbioru testowego.

Konsekwencja: pytanie badawcze brzmi „które cechy boiskowe budują wartość", a nie
„jak bardzo wycena jest lepka". To drugie jest znane i nieciekawe.

## D-06 — baseline „przepisz wycenę z t−1" jako punkt odniesienia, nie próg

Status: przyjęta

Baseline daje R² 0,8375 w skali logarytmicznej, MdAPE 30,0 procent i RMSLE 0,524.
Raportujemy go, ale nie traktujemy jako progu do pobicia.

Korzysta z informacji, której modele świadomie nie dostają (D-05), i nie istnieje dla
około 28 procent zbioru testowego. To inne zadanie z dostępem do innej informacji.

Konsekwencja raportowania: podajemy dwie liczby — na podzbiorze z dostępnym t−1 oraz
na pełnym teście z uzupełnieniem medianą grupową tam, gdzie t−1 brakuje. Podawanie
jednej z nich bez drugiej jest mylące.

## D-07 — odrzucenie bramkarzy

Status: przyjęta

Statystyki polowe są dla bramkarzy nieinformatywne. Modelowanie ich razem
z zawodnikami z pola wymagałoby osobnego zestawu cech i osobnego modelu. Tabela
`Goalkeeper` pozostaje nieużywana.

Koszt: praca nie obejmuje około 10 procent populacji zawodników. Do odnotowania
w ograniczeniach.

## D-08 — zakres: sześć sezonów, próg 225 minut

Status: przyjęta, próg parametryzowany

Sezony: 2017-2018, 2018-2019, 2020-2021, 2021-2022, 2022-2023, 2023-2024. Próg
minimalnych minut w sezonie: 225.

Odrzuciliśmy 2019-2020, bo Ligue 1 rozegrała 279 z 380 meczów; 2024-2025, bo ma
25 procent braków celu skorelowanych z ligą (La Liga 54,8 procent pokrycia wobec
Bundesligi 89,3); oraz 2025-2026, bo sezon trwa. Po odrzuceniu pokrycie wycen wynosi
od 99,1 do 99,6 procent w każdym sezonie.

Próg jest konieczny, bo poniżej niego statystyki per 90 minut są szumem: jedno
podanie w dziesięć minut daje dziewięć podań na dziewięćdziesiąt.

Koszt: próg usuwa około 19 procent par, i to nielosowo — wypadają młodzi, rezerwowi
i kontuzjowani, czyli systematycznie tańsi. Dlatego próg jest stałą w `parametry.py`,
a analiza wrażliwości dla wartości 0, 225, 450 i 900 jest zaplanowana.

## D-09 — braki zostają jako NaN, imputacja wewnątrz Pipeline

Status: przyjęta

Imputacja medianą liczona przed podziałem zbioru jest wyciekiem czasowym: mediana
zbioru testowego wpływa na wartości w treningu.

Pliki w `data/processed/` zachowują braki jako NaN. Imputacja odbywa się wewnątrz
`sklearn.Pipeline`, fitowana wyłącznie na zbiorze treningowym. To samo dotyczy
skalowania i selekcji korelacyjnej.

Konsekwencja: każdy kod używający tych plików musi przechodzić przez Pipeline.

Uwaga merytoryczna: braki we wskaźnikach procentowych są strukturalne — zawodnik nie
próbował danej akcji ani razu przez cały sezon. To jest informacja, nie brak.
Do rozważenia przy budowie Pipeline: wskaźnik z flagą „nie próbował" zamiast samej
imputacji.

## D-10 — kategoryczne jako tekst, OneHotEncoder bez drop

Status: przyjęta

`pozycja`, `region`, `liga` i `noga` przechowujemy jako tekst, kodowane przez
`OneHotEncoder(handle_unknown="ignore")` wewnątrz Pipeline.

Bezwzględnie bez `drop="first"`. `handle_unknown="ignore"` zamienia nieznaną kategorię
na wektor samych zer, i to jest poprawne. Dodanie `drop="first"` sprawia, że same zera
oznaczają kategorię referencyjną, więc Primeira Liga zostanie po cichu potraktowana
jak liga, którą enkoder usunął. Warte testu jednostkowego.

Uzasadnienie ogólne: `get_dummies` na całym zbiorze przed podziałem to wyciek plus
ryzyko rozjechania się zestawu kolumn między zbiorami.

## D-11 — liga zostaje cechą, z wariantem kontrolnym bez niej

Status: przyjęta

Primeira Liga jest kategorią nieobecną w treningu. Model nie ma fizycznej możliwości
poznać poziomu cen tej ligi — dostaje ją jako „żadna ze znanych lig".

Liga zostaje zmienną kategoryczną; nieznana kategoria jest częścią eksperymentu,
nie błędem. Dodatkowo każdy eksperyment przesunięcia dziedziny uruchamiamy w dwóch
wariantach. Bez ligi mierzymy degradację z samego przesunięcia relacji między cechami
a celem. Z ligą dokładamy koszt nieznanej kategorii. Różnica izoluje jedno od drugiego.
Bez tej kontroli nie wiadomo, czy mierzymy przesunięcie dziedziny, czy zachowanie
enkodera na wektorze spoza rozkładu treningowego.

Odrzucone: liga jako liczba, na przykład ranking UEFA. Dokładałaby zewnętrzne źródło
i ryzyko policzenia go na zbiorze testowym.

## D-12 — stratyfikacja „znani / nowi" jako obowiązkowa oś raportowania

Status: przyjęta

Podział jest czasowy, więc ten sam zawodnik występuje w wielu zbiorach w różnych
sezonach. Pary są rozłączne, więc formalnie wycieku nie ma, ale model rozpoznaje
zawodnika po kombinacji wzrostu, wieku, pozycji i profilu statystycznego, mimo że
`player_id` nie jest cechą.

Pomiar na HistGradientBoosting: na teście znani mają R² 0,68 wobec 0,56 dla nowych,
przy udziale znanych 66 procent. W zbiorze C znani mają 0,51 wobec −0,50 dla nowych,
przy udziale znanych 9 procent. Przewaga zostaje po wyrównaniu wieku grup, więc nie
jest artefaktem składu.

Podziału nie zmieniamy, bo w praktyce wycenia się zawodników, których rynek już zna.
Zamiast tego robimy trzy rzeczy. Etap 8 zapisuje kolumnę `znany_z_treningu`
w `model.parquet`. Każda metryka W1–W5 raportowana jest w trzech wariantach: całość,
znani, nowi. Porównanie testu ze zbiorem C w eksperymencie przesunięcia dziedziny
robimy na podzbiorach „nowi vs nowi", a wersja „całość vs całość" leci obok, jako
miara tego, ile z pozornej degradacji brało się ze składu próby.

Przy udziale znanych 66 procent wobec 9 procent porównanie jednej liczby na całym
teście z jedną liczbą na całym zbiorze C mierzy w dużej części różnicę składu, a nie
odporność modelu. Jest to najważniejszy eksperyment pracy, więc to nie jest analiza
dodatkowa, tylko warunek poprawności głównego wyniku.

## D-13 — rozkład degradacji na obciążenie i rozrzut

Status: przyjęta

Mediana wartości w Primeira Lidze wynosi 0,8 mln euro wobec 5,0 mln w Big 5. Jest to
fakt o poziomie sportowym i sile finansowej ligi, znany z góry, a nie anomalia. Model
trenowany na Big 5 będzie więc systematycznie i mocno zawyżał, niezależnie od tego,
jak dobrze rozumie grę — tym bardziej że Primeira Liga jest dla enkodera kategorią
nieznaną (D-11).

Degradację pod przesunięciem dziedziny raportujemy zawsze rozłożoną na dwa składniki:

```
RMSE² = (średnia reszty)² + (odchylenie std reszty)²
        └── obciążenie ──┘   └────── rozrzut ──────┘
```

Obciążenie kierunkowe to systematyczne mylenie się w jedną stronę o mniej więcej stałą
wartość; w skali logarytmicznej stałe przesunięcie oznacza stały mnożnik. Rozrzut to
błąd po odjęciu tej stałej, czyli odpowiedź na pytanie, czy model — znając już poziom
rynku — odróżnia lepszego zawodnika od gorszego.

Raportowanie samego RMSLE zmierzyłoby głównie fakt, że w Portugalii płacą mniej,
i nazwało go spadkiem odporności modelu. Dopiero rozrzut mówi cokolwiek o zdolnościach
modelu. Obie liczby mają też różne recepty: poziom naprawia się kilkunastoma etykietami
z nowej dziedziny, rozrzutu nie naprawi nic poza przetrenowaniem.

Rozszerzenie: warto zmierzyć wariant z rekalibracją przesunięcia na małej próbce
ze zbioru C i porównać go z wariantem bez rekalibracji. To daje praktyczną odpowiedź,
ile kosztuje wdrożenie modelu w nowej lidze.

## D-14 — reguła dostępu do zbioru C

Status: przyjęta, zastępuje regułę „zbiór zamknięty do końca"

Pierwotna reguła brzmiała: zbiór C otwierany raz, w eksperymencie końcowym, a każde
wcześniejsze zajrzenie unieważnia eksperyment. Reguła została złamana — przy audycie
danych policzono statystyki opisowe zbioru C, w tym rozkład zmiennej celu (średnia
log-wartości 13,93 wobec 15,70 na teście).

Nowa reguła jest zróżnicowana. Cechy zbioru C są dostępne wcześniej: pomiar
przesunięcia rozkładów cech przez PSI i test Kołmogorowa-Smirnowa to detekcja driftu
bez etykiet, dostępna w każdym wdrożeniu produkcyjnym. Cel zbioru C jest dostępny
tylko w eksperymencie końcowym — żadnego trenowania, strojenia, selekcji ani wyboru
metryki na tych danych.

Rozróżnienie jest merytorycznie uzasadnione i da się go obronić, w odróżnieniu
od reguły, którą już raz złamano. Wiedza o poziomie wycen w Portugalii nie daje
przewagi w modelowaniu, bo żaden model nie jest na tym zbiorze strojony; jest natomiast
przesłanką dla D-13, która poprawia jakość eksperymentu.

Zapisany fakt: przesunięcie w zbiorze C to w dominującej części przesunięcie poziomu
zmiennej celu, około 1,64 do 1,76 jednostki logarytmicznej, czyli około 1,3 odchylenia
standardowego. Nie jest to przesunięcie rozkładu cech — te są parytetowe, kolumny
licznikowe mają zero procent braków, a średnie xG wynoszą od 0,080 do 0,098 wobec
0,077 do 0,113 w Big 5.

## D-15 — metryki główne: RMSLE i MdAPE

Status: przyjęta

RMSLE i MdAPE jako główne, R² i MAE pomocniczo. MAPE wyłącznie do porównania
z wcześniejszymi podejściami, z jawną adnotacją o wadliwości.

MAPE dzieli przez wartość rzeczywistą, więc na skośnym rozkładzie mierzy głównie
zachowanie modelu na tanich zawodnikach. Jest też asymetryczna: zaniżenie ma sufit
stu procent, zawyżenie nie ma żadnego, więc model optymalizowany pod MAPE uczy się
systematycznie zaniżać. Pomiar na tej samej predykcji: MdAPE 30,0 procent, MAPE
41,3 procent.

## D-16 — podział czasowy z osobnym zbiorem kalibracyjnym

Status: przyjęta

Trening to sezony 2017-2018, 2018-2019, 2020-2021 i 2021-2022 w Big 5. Kalibracja
to 2022-2023 w Big 5. Test to 2023-2024 w Big 5. Zbiór C to Primeira Liga we wszystkich
dostępnych sezonach.

Podział losowy rozrzuciłby tego samego zawodnika po treningu i teście, a jego wycena
jest z roku na rok mocno skorelowana. Zbiór kalibracyjny jest wymagany przez W3:
kalibracja predykcji konforemnej na treningu daje zawyżone pokrycie i cały pomiar
traci sens.

Warunek: Primeira Liga nie może wystąpić w treningu ani kalibracji. Sprawdzane
asercją w etapie 8 i 10.

Otwarte: jeden sezon testowy to 1 957 par. Czy to wystarczy jako główny wynik, czy
przejść na rolling origin już w protokole walidacji zamiast dopiero w eksperymentach
odporności. Patrz pytanie O-1 w `04-plan.md`.

## D-17 — parquet, nie CSV

Status: przyjęta

Przy 114 kolumnach CSV gubi typy, a braki stają się pustymi ciągami nieodróżnialnymi
od zera. Parquet jest około pięć razy mniejszy i około dziesięć razy szybszy
w odczycie. Koszt: zależność `pyarrow`.

## D-18 — skrypty w pipelinie, notatniki tylko w eksploracji

Status: przyjęta

Pipeline danych to moduły `.py` w `case_study/football/`. Notatniki służą wyłącznie
do analizy eksploracyjnej i prezentacji wyników.

Notatnik zapisuje wyniki w pliku, więc każde uruchomienie robi ogromny diff w gicie.
Czytelna historia zmian ma bezpośrednie znaczenie dla W6.

Warunek dla notatników: `nbstripout` przed commitem, inaczej problem wraca.

## D-19 — separacja biblioteki od dziedziny

Status: przyjęta

`src/trustml/` nie wie, że istnieje piłka nożna. Przyjmuje strukturę problemu: X, y,
groups, time, domain, sensitive, strata, target\_scale. Specyfika dziedzinowa mieszka
w `case_study/football/`.

Testem przejścia jest uruchomienie na drugim, niepiłkarskim zbiorze bez zmiany ani
jednej linii w `src/`. Kandydat: ACSIncome z pakietu folktables. Zbiór nie ma struktury
grupowej, co jest zaletą testową: biblioteka musi zaraportować, że wyciek grupowy jest
niemierzalny, zamiast po cichu przejść na podział losowy.

Zasada budowy: moduł pakietu powstaje dopiero po ręcznym wykonaniu odpowiedniego
eksperymentu. Najpierw zrozum, potem uogólnij. Zabezpiecza przed rozrostem zakresu
pakietu kosztem części badawczej.

## D-20 — deduplikacja i validate w każdym merge

Status: przyjęta

Każda z siedmiu tabel polowych zawiera dokładnie dwa zduplikowane wpisy klucza
`(match_id, player_id)` na 521 261 wierszy — artefakt scrapingu. Bez walidacji te
wiersze rozmnażają się przez osiem kolejnych połączeń.

Deduplikujemy jawnie przed łączeniem, z raportem liczby usuniętych wierszy. Parametr
`validate=` stosujemy we wszystkich merge, bez wyjątków.

Wniosek ogólny: `validate` nie kosztuje nic i wykrywa klasę błędów, które inaczej
ujawniają się dopiero jako niewytłumaczalne wyniki modelu. Domyślnie, nie wyjątkowo.
