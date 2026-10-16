# Log decyzji

Format: D-xx: co zdecydowałem - dlaczego. Numery są używane w kodzie i w `configs/split.yaml`. Jeśli zmieniam decyzję, dopisuję nową i zaznaczam, którą zastępuje.

## Cel pracy i zakres

- **D-01:** Tematem jest sposób oceny wiarygodności modeli, a nie wybór najlepszego modelu - samo porównanie dokładności to mało do pracy magisterskiej. Główny wynik to macierz model x wymiar.
- **D-02:** Sześć wymiarów: W1 dokładność, W2 odporność, W3 niepewność, W4 wyjaśnialność, W5 sprawiedliwość, W6 odtwarzalność - wszystkie da się zmierzyć liczbowo w regresji. Prywatność i nadzór człowieka pomijam, bo tu się ich nie zmierzy.
- **D-07:** Bez bramkarzy - statystyki z pola nic o nich nie mówią, potrzebowaliby osobnego modelu.
- **D-08:** Sześć sezonów i próg 225 minut - poniżej progu statystyki na 90 minut to szum (1 podanie w 10 minut = 9 na 90). Odrzucone sezony i ich powody są w `docs/dane.md`; próg jest parametrem, żeby sprawdzić 0/450/900.
- **D-19:** `src/trustml/` nie wie nic o piłce, wszystko piłkarskie jest w `case_study/football/` - żeby bibliotekę dało się uruchomić na innym zbiorze bez zmian w `src/`.

## Cel i cechy

- **D-03:** Cel to `log1p(wartość w euro)` - rozkład wartości jest mocno skośny (3,6), po logarytmie prawie symetryczny.
- **D-04:** Cel nominalny, wersja zdeflowana tylko jako analiza wrażliwości - na HistGradientBoosting deflacja nic nie dała (MAE 0,576 wobec 0,573). Wrócę do tego, jeśli Elastic Net wyraźnie zyska na deflacji.
- **D-05:** Zeszłoroczna wycena nie jest cechą - to zbyt silny predyktor (sama wyjaśnia ok. 81% wariancji), przesłoniłby wszystko inne i ranking SHAP w W4 nie miałby sensu. Do tego to też wycena ludzi z Transfermarktu, a nie gra, i istnieje tylko dla ok. 72% zawodników z testu.
- **D-06:** Przepisanie zeszłorocznej wyceny (M0c) to punkt odniesienia, nie konkurent - korzysta z informacji, której modele nie dostają. Podaję je w dwóch wersjach: na zawodnikach z wyceną sprzed roku i na całym teście (braki z M0b).
- **D-09:** Braki zostają jako NaN, imputacja w Pipeline - imputacja przed podziałem przeniosłaby informację z testu do treningu.
- **D-10:** Kategorie jako tekst i `OneHotEncoder(handle_unknown="ignore")` bez `drop` - nieznana liga (Primeira Liga) daje same zera. Z `drop="first"` same zera oznaczałyby usuniętą kategorię, więc Primeira Liga po cichu udawałaby jedną z lig Big 5.
- **D-11:** Liga zostaje cechą, ale każdy eksperyment na zbiorze C robię też bez niej - różnica pokazuje, ile błędu wynika z samej nieznanej kategorii.
- **D-25:** Wyrzucam parę `dribble_success_percentage` i `tackled_perecentage` (O-6) - zmieniła definicję dokładnie na granicy trening/kalibracja (suma 100 w 98,7% wierszy przed i 21,1% po). Inne cechy ze skokiem poziomu zostają; koszt decyzji liczy `warianty.py`.
- **D-26:** Imputacja medianą z flagą braku (`add_indicator=True`) (O-3) - braki znaczą "zero prób" i są skupione u obrońców, więc sama mediana wprowadzałaby błąd zależny od pozycji.

## Podział i walidacja

- **D-12:** Wyniki raportuję osobno dla zawodników znanych z treningu i nowych - model rozpoznaje zawodnika po jego profilu (na teście R² 0,68 wobec 0,56). W zbiorze C nowych jest 91%, w teście 34%, więc porównanie bez tego podziału byłoby nieuczciwe.
- **D-13:** Błąd na zbiorze C rozkładam na przesunięcie (średnia reszty) i rozrzut - w Portugalii wszyscy są tańsi, więc model będzie zawyżał. Samo RMSLE mierzyłoby głównie to, a nie to, czy model odróżnia lepszych od gorszych.
- **D-14:** Ze zbioru C wolno mi wcześniej używać cech (PSI, test KS), cel dopiero w E11 - tak samo jest w produkcji: cechy nowych danych widać od razu, etykiet nie. Zastępuje regułę "zbiór C zamknięty do końca", którą złamałem przy audycie danych.
- **D-15:** Główne metryki to RMSLE i MdAPE, MAPE tylko do porównań - MAPE przy zaniżeniu ma sufit 100%, a przy zawyżeniu nie ma limitu, więc nagradza zaniżanie.
- **D-16:** Podział czasowy: trening 2017-22, kalibracja 2022-23, test 2023-24 - w praktyce przewiduje się do przodu, a przy podziale losowym ten sam zawodnik byłby w treningu i teście, czyli oczywisty wyciek. Do strojenia GroupKFold po zawodniku na treningu, z tego samego powodu.
- **D-22:** Wynik główny na jednym sezonie testowym (1 957 par), rolling origin dopiero w E7 (O-1) - inaczej każdy etap liczyłby się kilka razy dłużej. Różnice między modelami rzędu setnych R² trzeba podawać z przedziałem.
- **D-23:** M0b to mediana w komórce pozycja x liga, bez sezonu - sezonu testowego nie ma w treningu.

## Powrót do euro

- **D-21:** Przy kwotach w euro używam korekty Duana, liczonej tylko na resztach z treningu - `exp` ze średniej z logarytmów daje średnią geometryczną, a ta przy skośnym rozkładzie jest niższa od zwykłej średniej, więc kwoty byłyby zaniżone. Wzór: `exp(pred) * duan - 1`.
- **D-24:** Metryki (RMSLE, MdAPE, MAE) liczę bez korekty, korekta tylko przy sumach wycen (zastępuje część D-21) - te metryki są najmniejsze przy medianie, a korekta przesuwa predykcję w stronę średniej, więc tylko je pogarsza (sprawdziłem na wszystkich poziomach odniesienia). Dla M0c korekta szkodzi, więc zawsze podaję stosunek sum przed i po (`agregat`, `agregat_po_korekcie`).

## Narzędzia

- **D-17:** Parquet zamiast CSV - zachowuje typy i odróżnia brak od zera.
- **D-18:** Pipeline w skryptach `.py`, notatniki tylko do eksploracji - notatniki robią ogromne diffy w gicie.
- **D-20:** Usuwam duplikaty przed łączeniem i daję `validate=` w każdym merge - poprzednia wersja bez tego miała 846 zduplikowanych par zawodnik-sezon.
- **D-27:** Przebiegi zapisuję w repo (`e5_przebiegi.csv`, `e5_metryczka.json`), nie w MLflow - przy trzech modelach CSV wystarczy, a wyniki są w historii gita razem z kodem.
