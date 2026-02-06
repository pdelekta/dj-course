from gensim.models import Doc2Vec
from tokenizers import Tokenizer
from embedding_doc2vec_config import (
    TOKENIZER_FILE,
    OUTPUT_MODEL_FILE,
    files
)

# =========================================================================
# --- ETAP 1: Wczytanie, Tokenizacja i Przygotowanie Danych ---
# =========================================================================
try:
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku '{TOKENIZER_FILE}'. Upewnij się, że plik istnieje.")
    raise

try:
    print(f"Ładowanie modelu z pliku: {OUTPUT_MODEL_FILE}")
    model_d2v = Doc2Vec.load(OUTPUT_MODEL_FILE)
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku '{OUTPUT_MODEL_FILE}'. Uruchom run-cbow.py najpierw.")
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

# =========================================================================
# === ETAP 2: BEZPOŚREDNIE WNIOSKOWANIE (INFERENCE)
# =========================================================================

print("\n" + "="*50)
print("=== ROZPOCZYNAM ETAP WNIOSKOWANIA (INFERENCE) ===")
print("="*50)

#  🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥testowanie🔥🔥🔥🔥🔥🔥🔥🔥
new_sentence = "Jestem głodny i bardzo chętnie zjadłbym coś."
print(f"Zdanie do wnioskowania: \"{new_sentence}\"")


# Używamy obiektów już załadowanych/wytrenowanych: model_d2v, tokenizer, raw_sentences
loaded_model = model_d2v # Używamy modelu prosto z treningu
sentence_lookup = raw_sentences # Używamy listy zdań prosto z wczytywania korpusu


# Tokenizacja nowego zdania
new_tokens = tokenizer.encode(new_sentence).tokens

# 2. Generowanie wektora dla nowego zdania
inferred_vector = loaded_model.infer_vector(new_tokens, epochs=loaded_model.epochs)
print(f"\nWygenerowany wektor (embedding) dla zdania. Kształt: {inferred_vector.shape}")

# 3. Znajdowanie najbardziej podobnych wektorów z przestrzeni dokumentów/zdań
# topn=5, topn=20 - za co odpowiadaten parametr?
most_similar_docs = loaded_model.dv.most_similar([inferred_vector], topn=5)

print("\n5 najbardziej podobnych zdań z korpusu (Doc2Vec Inference):")
for doc_id_str, similarity in most_similar_docs:
    # 1. Konwertujemy ID (string) z powrotem na indeks (int)
    doc_index = int(doc_id_str)

    # 2. Używamy indeksu do odnalezienia oryginalnego tekstu
    # Zabezpieczenie na wypadek błędu indeksowania (choć nie powinno wystąpić)
    try:
        original_sentence = sentence_lookup[doc_index]
        print(f"  - Sim: {similarity:.4f} | Zdanie (ID: {doc_id_str}): {original_sentence}")
    except IndexError:
         print(f"  - Sim: {similarity:.4f} | BŁĄD: Nie znaleziono zdania dla ID: {doc_id_str}")

print("\n=== ETAP WNIOSKOWANIA ZAKOŃCZONY ===")
