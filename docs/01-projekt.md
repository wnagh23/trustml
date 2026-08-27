# Projekt

Wersja 1.1, 2026-08-27

## Co budujemy

Produktem pracy jest biblioteka `trustml` — pakiet Pythona, który przyjmuje dowolny
model zgodny z API scikit-learn i produkuje wystandaryzowany raport wiarygodności
w sześciu wymiarach.

Predykcja wartości rynkowej piłkarzy jest przykładem demonstracyjnym dla tej
biblioteki, nie celem samym w sobie. Żyje w `case_study/football/` i biblioteka
nic o niej nie wie.

Temat pracy: „Ocena wiarygodności modeli uczenia maszynowego — wielowymiarowy
framework ewaluacji na przykładzie predykcji wartości rynkowej piłkarzy".

## Założenie robocze

Rankingi modeli według różnych wymiarów wiarygodności są rozbieżne. Model najlepszy
pod względem dokładności nie jest najlepszy pod względem odporności, kalibracji
niepewności czy stabilności wyjaśnień. Gdyby było inaczej, cały framework byłby
zbędny i wystarczyłoby R².

Weryfikacja tego założenia — liczbowo, przez korelację rangową Kendalla między
rankingami z różnych wymiarów — jest głównym wynikiem części eksperymentalnej.
Głównym artefaktem jest macierz model × wymiar, nie tabela z dokładnością.

## Sześć wymiarów

Zasada porządkowa: przy każdym użyciu oznaczenia W1–W6, w rozmowie, w kodzie
i w dokumentacji, dopisujemy krótkie wyjaśnienie, jaka metryka za nim stoi.
Nie zakładamy, że skrót jest oczywisty.

### W1 — poprawność predykcyjna

Jak blisko prawdy są predykcje.

Metryką główną jest RMSLE, czyli pierwiastek ze średniej kwadratów błędu
logarytmicznego. Karze błąd względny, nie bezwzględny: pomyłka o czynnik dwa
kosztuje tyle samo przy zawodniku za pół miliona i za pięćdziesiąt. Model trenuje
na logarytmie celu ze stratą kwadratową, więc funkcja celu treningu i metryka
ewaluacji są tym samym obiektem.

Obok niej MdAPE — mediana bezwzględnego błędu procentowego. To metryka
komunikacyjna: „typowo mylimy się o X procent". Mediana, a nie średnia, bo rozkład
błędów procentowych ma długi prawy ogon. R² w skali logarytmicznej służy jako
punkt porównawczy.

MAPE nie jest metryką główną i nie będzie. Rozkład celu jest skośny, więc średnia
jest zdominowana przez ogon. Metryka jest też asymetryczna: zaniżenie ma sufit
stu procent, zawyżenie nie ma żadnego, więc model optymalizowany pod MAPE uczy się
systematycznie zaniżać. Pomiar na tej samej predykcji: MdAPE 30,0 procent,
MAPE 41,3 procent.

### W2 — odporność

O ile pogarsza się model, gdy warunki odbiegają od treningowych.

Podstawowy pomiar to degradacja między zbiorem testowym (Big 5) a zbiorem C
(Primeira Liga). Degradację raportujemy zawsze rozłożoną na dwa składniki:

```
RMSE² = (średnia reszty)² + (odchylenie std reszty)²
        └── obciążenie ──┘   └────── rozrzut ──────┘
```

Obciążenie kierunkowe to systematyczne mylenie się w jedną stronę o mniej więcej
stałą wartość; w skali logarytmicznej stałe przesunięcie oznacza stały mnożnik.
Rozrzut to błąd po odjęciu tej stałej, czyli odpowiedź na pytanie, czy model —
znając już poziom rynku — odróżnia lepszego zawodnika od gorszego. Szczegóły
i uzasadnienie w decyzji D-13.

Poza tym mierzymy wrażliwość na szum (szum gaussowski o zadanym odchyleniu na
cechach `_p90`, przyrost RMSLE jako funkcja siły szumu; metryka zbiorcza to
nachylenie krzywej degradacji) oraz stabilność przy usunięciu pojedynczej cechy.
Model opierający się na jednej cesze jest kruchy.

Każdy eksperyment przesunięcia dziedziny uruchamiamy w dwóch wariantach: z ligą
jako cechą i bez niej. Wariant bez ligi pokazuje degradację z samego przesunięcia
relacji między cechami a celem, wariant z ligą dokłada koszt nieznanej kategorii,
a różnica izoluje jedno od drugiego. Bez tej kontroli nie wiadomo, czy mierzymy
przesunięcie dziedziny, czy zachowanie enkodera na wektorze spoza rozkładu
treningowego.

### W3 — niepewność

Czy model wie, kiedy nie wie.

Mierzymy pokrycie empiryczne przedziałów konforemnych: dla poziomu ufności 1−α
sprawdzamy, jaki odsetek prawdziwych wartości faktycznie wpada w przedział. Metoda
działa poprawnie, gdy pokrycie empiryczne odpowiada nominalnemu. Drugą liczbą jest
średnia szerokość przedziału — przy tym samym pokryciu węższy jest lepszy. Te dwie
liczby raportuje się zawsze razem, bo każda osobno nic nie znaczy.

Narzędzie: MAPIE. Zbiór kalibracyjny (sezon 2022-2023) istnieje wyłącznie po to;
kalibracja na treningu daje zawyżone pokrycie i cały pomiar traci sens.

Wyróżnikiem pracy jest pomiar pokrycia pod przesunięciem dziedziny. Predykcja
konforemna zakłada wymienialność próbek, a przesunięcie ją łamie, więc pokrycie
powinno spaść w przewidywalny sposób. To jest hipoteza stawiana przed
eksperymentem, nie obserwacja po fakcie.

Drabinka lig — mediana wartości od 0,8 mln euro w Primeira Lidze przez 4–5 mln
w Ligue 1, Serie A i La Liga, 5 mln w Bundeslidze, po 15 mln w Premier League —
pozwala zbudować krzywą dawka-odpowiedź zamiast pojedynczego testu binarnego.

### W4 — wyjaśnialność

Czy wiadomo dlaczego, i czy to wyjaśnienie jest powtarzalne.

Wartości SHAP dają wkład każdej cechy w konkretną predykcję oraz ranking ważności
jako średnią z wartości bezwzględnych. Sam ranking to jednak dopiero połowa
odpowiedzi.

Drugą połową, i wyróżnikiem pracy, jest korelacja rangowa Kendalla. Ranking ważności
budujemy osobno dla kilku ziaren losowych i kilku foldów, a potem liczymy tau
między każdą parą rankingów. Tau bliskie jedności oznacza, że wyjaśnienia są
stabilne i wolno je interpretować. Tau niskie oznacza, że ranking jest w dużej
mierze losowy, a interpretacja pojedynczego przebiegu byłaby nadinterpretacją.
Raportujemy rozkład tau po parach, nie pojedynczą liczbę.

Kryterium jest ostre: model o niestabilnych wyjaśnieniach jest bezużyteczny
w zastosowaniu decyzyjnym niezależnie od tego, jak wysokie ma R². Skaut nie może
dostać dwóch różnych uzasadnień tej samej wyceny.

### W5 — sprawiedliwość

Czy model działa równie dobrze dla wyróżnionych grup. W regresji sprowadza się to
do dwóch niezależnych pomiarów.

Parytet jakości pyta, czy model jest tak samo dokładny w każdej grupie — mierzymy
różnicę MdAPE lub RMSLE między grupami. Parytet obciążenia pyta, czy model
systematycznie zaniża lub zawyża w danej grupie — mierzymy średnią reszty w skali
logarytmicznej, osobno per grupa.

Te pomiary są niezależne. Model może być równie dokładny dla wszystkich grup,
a mimo to systematycznie przesunięty dla jednej z nich.

Cechy wrażliwe w tym projekcie: region, wiek, pozycja, liga. Narzędzie: fairlearn.

Jedno ograniczenie warte akapitu w pracy, ale nie więcej. Cel pochodzi z szacunków
społeczności Transfermarkt, nie z cen transakcyjnych. Model wiernie odtwarzający
rynek pokaże zerową dysproporcję błędu, powielając jednocześnie ewentualne
uprzedzenia rynku. W5 mierzone wobec tego celu odpowiada na pytanie o sprawiedliwość
wobec rynku, a nie o sprawiedliwość jako taką.

### W6 — odtwarzalność

Czy ten sam wynik da się uzyskać ponownie.

Pierwszy pomiar to rozrzut metryk między ziarnami losowymi — odchylenie standardowe
RMSLE po n uruchomieniach. Duży rozrzut oznacza, że pojedynczy raportowany wynik
jest przypadkiem, więc wyniki wrażliwe na ziarno raportujemy jako rozkład, nie
jako punkt.

Drugi to kompletność zapisu przebiegu: wersje pakietów, ziarno, suma kontrolna
danych wejściowych, hash commita, parametry. Przebieg bez tego nie jest odtwarzalny
niezależnie od tego, jak dobre ma metryki.

Narzędzie: MLflow.

Stan faktyczny jest tu słaby. Prowieniencja danych surowych jest zerowa — `master.db`
nie zawiera żadnych metadanych scrape'u. To największy dług tego wymiaru, opisany
w `02-dane.md`.

## Modele

Poziom odniesienia M0 składa się z trzech wariantów: mediana globalna log-wartości
(M0a), mediana w komórce pozycja × liga × sezon (M0b) oraz przepisanie wyceny
z sezonu poprzedniego (M0c).

M1 to Elastic Net na `log1p(y)`. Zwykły OLS odpada przy około stu cechach i silnej
współliniowości statystyk piłkarskich. Elastic Net obejmuje Ridge i Lasso jako
przypadki brzegowe, więc wybór między nimi jest wynikiem walidacji krzyżowej,
a nie arbitralną decyzją. Stabilność współczynników jest w tej pracy przedmiotem
badania w W4, więc model bazowy nie może być z góry niestabilny.

M2 to Random Forest. Ma inny profil bias-variance niż boosting, a rozrzut predykcji
poszczególnych drzew daje darmowy sygnał niepewności do W3.

M3 to XGBoost — model docelowy, zwykle najlepszy w W1.

Wszystkie przechodzą identyczną procedurę ewaluacji W1–W6.

Sieci neuronowe odrzucone: na danych tabelarycznych tej wielkości nie dają przewagi,
a utrudniają wyjaśnialność. LightGBM i CatBoost pozostają opcjonalnym sprawdzeniem,
czy wnioski o W2–W5 nie są artefaktem konkretnej implementacji boostingu.

Uwaga do M0c. Baseline „przepisz zeszłoroczną wycenę" daje R² 0,8375 w skali
logarytmicznej, MdAPE 30,0 procent i RMSLE 0,524. Nie jest to jednak próg do pobicia:
korzysta z informacji, której modele świadomie nie dostają (D-05), i nie istnieje
dla około 28 procent zbioru testowego. To inne zadanie z dostępem do innej
informacji — punkt odniesienia, nie konkurent.

## Artefakt: pakiet trustml

Docelowe API:

```python
from trustml import TrustReport

report = TrustReport(
    model=pipeline,
    X_train=..., y_train=...,
    X_calib=..., y_calib=...,   # do predykcji konforemnej (W3)
    X_test=..., y_test=...,
    X_shift=..., y_shift=...,   # inna dziedzina (W2)
    groups=...,                 # klucz grupowania, np. player_id
    strata=...,                 # osie stratyfikacji, np. znany/nowy
    sensitive_features=...,     # do W5
)
report.run().to_html("reports/xgboost.html")
```

Zakres wersji 0.1 to W1, W6 i eksport HTML. Wymiary W2–W5 dokładamy iteracyjnie.

Zasada budowy: moduł pakietu powstaje dopiero po ręcznym wykonaniu odpowiedniego
eksperymentu. Najpierw zrozum, potem uogólnij. Zabezpiecza to przed rozrostem
zakresu pakietu kosztem części badawczej.

### Separacja od dziedziny

`src/trustml/` nie wie, że istnieje piłka nożna. Przyjmuje strukturę problemu:
X, y, groups, time, domain, sensitive, strata, target_scale. Specyfika dziedzinowa —
crosswalk, normalizacja per 90 minut, deflacja, okno wyceny — mieszka
w `case_study/football/`.

Testem przejścia jest uruchomienie na drugim, niepiłkarskim zbiorze bez zmiany ani
jednej linii w `src/`. Kandydat: ACSIncome z pakietu folktables (Ding i in.,
NeurIPS 2021) — dochód jako cel skośny, stan USA jako dziedzina, rok spisu jako czas,
cechy wrażliwe gotowe. Zbiór nie ma struktury grupowej, co jest zaletą testową:
biblioteka musi zaraportować, że wyciek grupowy jest niemierzalny, zamiast po cichu
przejść na podział losowy.

Narzędzie do oceny wiarygodności, które ukrywa własne ograniczenia, jest wewnętrznie
sprzeczne.

### Do rozstrzygnięcia

Czym `trustml` różni się od deepchecks, giskard, evidently i fairlearn. Bez
odpowiedzi wkład pracy jest podważalny na obronie. Hipoteza do zweryfikowania przez
lekturę ich dokumentacji, nie przez zgadywanie: tamte narzędzia walidują i monitorują
pojedynczy model, a tutaj jednostką analizy jest macierz model × wymiar
i rozbieżność rankingów.

## Zasady pracy

Zbiór C (Primeira Liga) jest zamknięty do eksperymentu końcowego, ale reguła jest
zróżnicowana. Pomiar przesunięcia cech — PSI, test Kołmogorowa-Smirnowa — jest
dozwolony wcześniej, bo to detekcja driftu bez etykiet, dostępna w każdym wdrożeniu
produkcyjnym. Dotykanie zmiennej celu w zbiorze C poza eksperymentem końcowym nie
jest dozwolone. Szczegóły w D-14.

Podział temporalny jest twardy i żaden krok preprocessingu nie widzi testu.
Wszystko, co uczy się z danych — imputacja, skalowanie, selekcja korelacyjna,
kodowanie kategorii — żyje wewnątrz `sklearn.Pipeline`.

Grupujemy po `player_id`, nigdy po nazwisku.

Każda metryka raportowana jest w rozbiciu na zawodników znanych i nieznanych
z treningu (D-12).

Każda liczba, która trafia do wniosków, pochodzi z uruchamialnego kodu
w repozytorium. Ziarno losowe jest ustawione i zapisane w konfiguracji.

Kod komentujemy na poziomie pozwalającym zrozumieć każdy krok bez czytania
dokumentacji bibliotek. Nazwy w kodzie po polsku, bez diakrytyków; nazwy kolumn
ze źródeł zostają oryginalne.

Notatniki służą wyłącznie eksploracji i prezentacji. Logika mieszka w modułach.

## Powiązane dokumenty

- `02-dane.md` — źródła, pipeline, słownik zbioru, ograniczenia
- `03-decyzje.md` — log decyzji projektowych
- `04-plan.md` — mapa drogowa i status
