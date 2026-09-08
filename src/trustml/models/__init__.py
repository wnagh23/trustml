"""
Budowa modeli: wspolne przetwarzanie cech, niezalezne od dziedziny.

Definicje samych modeli i przestrzenie hiperparametrow mieszkaja po stronie
zastosowania. Zgodnie z zasada z D-19 modul biblioteki powstaje dopiero po
recznym wykonaniu eksperymentu - najpierw zrozum, potem uogolnij.
"""

from .pipeline import FiltrKorelacyjny, zbuduj_model, zbuduj_przetwarzanie

__all__ = ["FiltrKorelacyjny", "zbuduj_model", "zbuduj_przetwarzanie"]
