import numpy as np
import json
import logging
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tokenizers import Tokenizer
import os
import glob
import time
from embedding_doc2vec_config import (
    TOKENIZER_FILE,
    MIN_COUNT,
    OUTPUT_MODEL_FILE,
    VECTOR_LENGTH,
    WINDOW_SIZE,
    WORKERS,
    EPOCHS,
    OUTPUT_SENTENCE_MAP,
    files
)

# Ustawienie logowania dla gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- ETAP 1: Wczytanie, Tokenizacja i Przygotowanie Danych ---
try:
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku '{TOKENIZER_FILE}'. Upewnij się, że plik istnieje.")
    raise

# Wczytywanie i agregacja tekstu
raw_sentences = []
print("Wczytywanie tekstu z plików...")
print(f"Liczba plików do wczytania: {len(files)}")

for file in files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            raw_sentences.extend(lines)
    except FileNotFoundError:
        print(f"OSTRZEŻENIE: Nie znaleziono pliku '{file}'. Pomijam.")
        continue
    except Exception as e:
        print(f"BŁĄD podczas przetwarzania pliku '{file}': {e}")
        continue

if not raw_sentences:
    print("BŁĄD: Korpus danych jest pusty.")
    raise ValueError("Korpus danych jest pusty.")
print(f"Tokenizacja {len(raw_sentences)} zdań...")

# Konwersja na listę tokenów
tokenized_sentences = [
    tokenizer.encode(sentence).tokens for sentence in raw_sentences
]

# Przygotowanie danych dla Doc2Vec
tagged_data = [
    TaggedDocument(words=tokenized_sentences[i], tags=[str(i)])
    for i in range(len(tokenized_sentences))
]
print(f"Przygotowano {len(tagged_data)} sekwencji TaggedDocument do treningu.")

# --- ETAP 2: Trening Doc2Vec ---
print("\n--- Rozpoczynanie Treningu Doc2Vec ---")
start_time = time.time()
model_d2v = Doc2Vec(
    tagged_data,
    vector_size=VECTOR_LENGTH,
    window=WINDOW_SIZE,
    min_count=MIN_COUNT,
    workers=WORKERS,
    epochs=EPOCHS,
    dm=1 # Distributed Memory (PV-DM)
)
end_time = time.time()
print(f"Trening zakończony pomyślnie. Czas trwania: {end_time - start_time:.2f}s")

# --- ETAP 3: Zapisywanie Wytrenowanego Modelu i Mapy ---
try:
    model_d2v.save(OUTPUT_MODEL_FILE)
    print(f"\nPełny model Doc2Vec zapisany jako: '{OUTPUT_MODEL_FILE}'.")

    with open(OUTPUT_SENTENCE_MAP, "w", encoding="utf-8") as f:
        json.dump(raw_sentences, f, ensure_ascii=False, indent=4)
    print(f"Mapa zdań do ID zapisana jako: '{OUTPUT_SENTENCE_MAP}'.")

except Exception as e:
    # W kontekście 'połączonego skryptu' błąd zapisu nie przerywa wnioskowania
    print(f"OSTRZEŻENIE: BŁĄD podczas zapisu modelu/mapy: {e}. Kontynuuję wnioskowanie in-memory.")