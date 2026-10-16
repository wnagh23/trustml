# Dane

Źródła, pipeline, zbiór modelowy i najważniejsze wnioski z EDA. Liczby z `model.parquet` z 2026-08-24.

## Źródła

**FBref**: `data/raw/fbref/master.db` (SQLite, ok. 211 MB). Statystyki mecz po meczu, klucz `(match_id, player_id)`.

| tabela | wiersze | co zawiera |
|---|---|---|
| Match | 17 551 | sezon, data, drużyny, wynik |
| Player_Info | 521 261 | nazwisko, narodowość, pozycja, wiek, minuty |
| Summary, Passing, Possession, Defensive_Actions, Pass_Types, Miscellaneous | po 521 261 | statystyki gry |
| Goalkeeper | 35 510 | nie używam, bramkarzy odrzucam (D-07) |

**Transfermarkt**: zbiór Kaggle `davidcariboo/player-scores` w `data/raw/tm/`. Używam `players.csv` (data urodzenia, wzrost, noga) i `player_valuations.csv` (wyceny z datami).

`master.db` nie ma żadnych metadanych. Pochodzenie odtworzyłem z archiwów ZIP ze stronami FBref (`scripts/audyt_zrodel.py`, wynik w `docs/zrodla_fbref.json`). Liczba stron zgadza się z liczbą meczów w bazie co do jednego.

Do sprawdzenia przed oddaniem pracy: warunki FBref dotyczące scrapingu i licencja zbioru z Kaggle.

## Pipeline

`python -m case_study.football.przygotuj_dane`

| etap | moduł | co robi |
|---|---|---|
| 1-2 | `zrodla.py` | wczytanie tabel, usunięcie duplikatów, połączenie po kluczu |
| 3 | `cechy.py` | wiek, narodowość, pozycja |
| 4 | `cechy.py` | agregacja mecz -> zawodnik x sezon, próg 225 minut |
| 5 | `cechy.py` | wskaźniki procentowe, statystyki na 90 minut |
| 6 | `wycena.py` | połączenie z Transfermarktem, wzrost i noga |
| 7 | `wycena.py` | wycena z okna kwiecień-wrzesień |
| 8 | `zbiory.py` | podział na zbiory |
| 9 | `zbiory.py` | deflacja |
| 10 | `przygotuj_dane.py` | zapis plików i `manifest.json` |

Parametry są w `parametry.py`. Każde łączenie ma `validate=`. Poprzednia wersja pipeline'u łączyła po nazwisku bez walidacji i miała przez to 846 zduplikowanych par zawodnik-sezon.

Liczba wierszy po każdym etapie trafia do `manifest.json` razem z SHA-256 plików wejściowych, wersjami bibliotek i parametrami. Dzięki temu widać, czy dwa przebiegi pracowały na tych samych danych.

## Najważniejsze reguły

**Procenty.** Liczę je jako suma udanych / suma prób w sezonie. Średnia z procentów z meczów jest błędna: 1/1 w jednym meczu i 1/10 w drugim daje średnio 55%, a naprawdę jest 2/11, czyli 18%. Na tych danych naiwna średnia myliła się nawet o 33 p.p.

**Per 90 minut.** Statystyki licznikowe dzielę przez `minuty / 90`. Inaczej model uczyłby się głównie tego, ile ktoś grał. Minuty i mecze zostają jako osobne cechy.

**Pozycja.** Biorę pierwszy kod z pola pozycji (`FW,AM` to napastnik, `AM,FW` pomocnik). Potem w sezonie wybieram pozycję z największą liczbą minut. Trzy klasy: DEF, MID, FOR.

**Region.** EUROPA (według UEFA), AMERYKA_PLD (CONMEBOL) i RESZTA. Surowa narodowość miałaby 126 wartości.

**Wiek.** FBref podaje wiek jako `lata-dni` w dniu meczu. Z tego liczę datę urodzenia, a potem wiek na 30 czerwca. Dodaję kwadrat wieku, żeby model liniowy mógł złapać parabolę.

**Łączenie z Transfermarktem.** Trzy kroki: nazwisko + data urodzenia, potem data urodzenia + kraj, na końcu samo nazwisko. Nazwiska powtarzające się po którejś stronie pomijam, żeby nie przypisać komuś cudzej wyceny. Odrzucam pary z datą urodzenia różną o więcej niż 7 dni. Zostaje 13 761 z 14 080 par (97,7%).

**Moment wyceny.** Dla sezonu kończącego się w roku t biorę wycenę z okna 1 kwietnia - 30 września, najbliższą 30 czerwca. To moment tuż po sezonie, więc wycena najlepiej odpowiada temu, jak zawodnik zagrał. Nie biorę maksimum z okna, bo byłoby zawyżone, i to bardziej u zawodników wycenianych częściej.

**Deflacja.** Indeks łańcuchowy: dla kolejnych sezonów porównuję średni log wartości w komórkach wiek x pozycja. Liczę go tylko na treningu. Kalibracja, test i zbiór C dostają współczynnik ostatniego sezonu treningowego. Wersja na stałym panelu (ci sami zawodnicy co roku) była błędna, bo mieszała inflację ze starzeniem się zawodników. Inflacja w Big 5 między 2017-18 a 2023-24 to ok. 38%.

## Zbiór modelowy

`data/processed/model.parquet`: 13 709 wierszy x 115 kolumn, 5 310 zawodników, klucz `(player_id, season)`.

| zbiór | wiersze | sezony | ligi |
|---|---|---|---|
| trening | 7 924 | 2017-18, 2018-19, 2020-21, 2021-22 | Big 5 |
| kalibracja | 1 982 | 2022-23 | Big 5 |
| test | 1 957 | 2023-24 | Big 5 |
| zbiór C | 1 846 | 2018-19 do 2023-24 | Primeira Liga |

Odrzucone sezony: 2019-20 (Ligue 1 rozegrała 279 z 380 meczów), 2024-25 (25% braków wyceny, różnie w ligach), 2025-26 (trwa).

Kolumny: 6 kolumn klucza i metadanych (w tym `podzial` i `znany_z_treningu`), 4 wersje celu i 105 cech: 87 statystyk `_p90`, 8 wskaźników procentowych, `wiek`, `wiek_do_kw`, `wzrost_cm`, `minuty_sezon`, `mecze` oraz 5 kategorycznych (`pozycja`, `region`, `liga`, `noga`, `zmienil_lige`). W modelach używam 103 cech, bo wyrzuciłem parę wskaźników dryblingu (D-25).

Braki są małe (maks. 2,2%) i wynikają z zera prób w mianowniku. Zostają jako NaN, a imputację robi Pipeline (D-09, D-26).

## Wnioski z EDA

Kod: `case_study/football/eda.py`, uruchomienie `python -m case_study.football.uruchom_eda`. Wykresy w `reports/figures/eda/`, liczby w `reports/tables/eda/fakty.json`. Jakość modeli w EDA liczyłem na kalibracji, żeby nie dotykać testu.

1. **Zmiana definicji dryblingu.** Do sezonu 2021-22 `dribble_success_percentage` i `tackled_perecentage` sumują się do 100 w 98,7% wierszy. Od 2022-23 tylko w 21,1%. Zmiana wypada dokładnie na granicy trening/kalibracja, więc wyrzuciłem tę parę (D-25).
2. **Liga.** Zawodnik Premier League jest wart ok. 2,1 razy więcej niż porównywalny zawodnik (ten sam wiek, pozycja i sezon). Ligue 1 ok. 0,7 razy.
3. **Region.** Po kontroli wieku, pozycji, ligi i sezonu zawodnicy z Ameryki Południowej są wyceniani ok. 40% wyżej, a reszta świata ok. 19% niżej niż Europejczycy. To dotyczy wycen rynku, nie modelu, ale będzie ważne w W5.
4. **Wzrost.** Ogólna korelacja wzrostu z wartością to ok. 0. W pozycjach znak się zmienia: u napastników -0,09, u obrońców +0,08, u pomocników +0,05 (paradoks Simpsona). Efekty są słabe. Może warto sprawdzić model osobno dla pozycji.
5. **Minuty ważniejsze niż statystyki.** Minuty, mecze i wzrost dokładają 0,177 R² do samego kontekstu, a wszystkie 95 statystyk gry jeszcze tylko 0,092. Minuty są jednak skutkiem wartości tak samo jak przyczyną, więc nie można ich czytać przyczynowo.
6. **Sufit.** Sam kontekst (wiek, pozycja, liga, region, noga) daje R² 0,31 na kalibracji. HistGradientBoosting bez strojenia na wszystkich cechach daje 0,72.
7. **18-latkowie najdrożsi.** Mediana 10 mln euro wobec 8,25 mln u 25-latków. To efekt progu 225 minut, który przepuszcza tylko młodych grających w Big 5.
8. **Braki skupione w pozycji.** Braki we wskaźnikach dryblingu: 4,9% u obrońców, 0,2% u napastników. Dlatego imputacja ma flagę braku (D-26).
9. **Cel jest ziarnisty.** Transfermarkt wycenia w okrągłych progach: 119 różnych wartości, 10 najczęstszych to 45% obserwacji. Część błędu to po prostu zaokrąglenie.
10. **Przesunięcie cech do zbioru C jest małe.** Mediana PSI 0,033, tylko 2 cechy powyżej 0,25. Zbiór C różni się głównie poziomem cen (ok. 1,7 w skali log), więc w E11 błąd będzie głównie przesunięciem w jedną stronę (D-13).

## Ograniczenia

- Cel to opinia społeczności Transfermarkt, nie cena transferu. Model odtwarza tę opinię razem z jej błędami.
- Brak bramkarzy (ok. 10% zawodników).
- Próg 225 minut usuwa ok. 19% par, głównie młodych, rezerwowych i kontuzjowanych.
- Ten sam zawodnik bywa w treningu i w teście (różne sezony). Model go "rozpoznaje", więc wyniki podaję osobno dla znanych i nowych (D-12).
- 153 zawodników występuje w Big 5 i w Primeira Lidze, w różnych sezonach.
- Brak danych o popularności (próby z Wikipedią i Google Trends się nie udały).
- Dane osobowe są publiczne, ale przetworzone pliki trzymam poza repo.
