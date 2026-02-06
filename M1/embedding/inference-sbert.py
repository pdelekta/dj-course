import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import time
from run_sbert import load_raw_sentences
from embedding_sbert_config import (
    MODEL_NAME,
    OUTPUT_EMBEDDINGS_FILE,
    files
)

try:
    raw_sentences = load_raw_sentences(files)
    print(f"Wczytano {len(raw_sentences)} zdań do przetworzenia.")
except ValueError as e:
    print(f"BŁĄD: {e}")
    exit()

# Sprawdzenie, czy wektory korpusu istnieją już na dysku
if os.path.exists(OUTPUT_EMBEDDINGS_FILE):
    print(f"\n--- Wariant 1: Wczytywanie wektorów z pliku '{OUTPUT_EMBEDDINGS_FILE}' ---")
    try:
        start_time = time.time()
        sentence_embeddings = np.load(OUTPUT_EMBEDDINGS_FILE)
        end_time = time.time()
        print(f"Wektory załadowane pomyślnie w {end_time - start_time:.2f} sekundy. Pominięto kodowanie.")

    except Exception as e:
        # W przypadku błędu wczytywania (np. uszkodzony plik), przejdź do generowania
        print(f"BŁĄD podczas ładowania pliku .npy: {e}. Przetwarzam korpus od nowa.")

# --- ETAP 3: Przykładowe Wykorzystanie (Porównywanie Zdań) ---

# =========================================================
# === DODANY FRAGMENT KODU ROZWIĄZUJĄCY BŁĄD NameError ===
# =========================================================
# Sprawdzenie, czy model został już zainicjowany (tj. czy zmienna istnieje w globalnym zakresie)
if 'model_sbert' not in locals() and 'model_sbert' not in globals():
    print(f"\nŁadowanie Sentence-Transformer do kodowania zapytania: {MODEL_NAME}...")
    try:
        model_sbert = SentenceTransformer(MODEL_NAME)
        print("Model SBERT załadowany pomyślnie.")
    except Exception as e:
        print(f"BŁĄD podczas ładowania modelu dla zapytania: {e}")
        exit()
# =========================================================
# 🔥🔥🔥🔥🔥🔥testowanie zdań🔥🔥🔥🔥🔥🔥🔥🔥
query_sentence = "Spędzili miło wieczór i poszli spać."
# query_sentence = "Wojsko wejdzie do miast i skończą się bunty"
# query_sentence = "Leczenie tego schorzenia jest bardzo ważne i wymaga interwencji lekarza."
print(f"\n--- Wyszukiwanie podobieństwa do: '{query_sentence}' ---")

# Generowanie wektora dla zapytania
query_embedding = model_sbert.encode(
    [query_sentence],
    convert_to_numpy=True
)

# Obliczenie podobieństwa kosinusowego między zapytaniem a wszystkimi zdaniami
# Podobieństwo kosinusowe jest standardową miarą podobieństwa wektorów
similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]

# Wyszukanie 10 najbardziej podobnych
top_10_indices = np.argsort(similarities)[::-1][:10]

print("\n5 zdań z korpusu najbardziej podobnych do zapytania:")
for i in top_10_indices:
    print(f"  - Sim: {similarities[i]:.4f} | Zdanie: {raw_sentences[i]}")