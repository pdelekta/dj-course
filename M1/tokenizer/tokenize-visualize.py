import os
from pathlib import Path
from tokenizers import Tokenizer
from typing import TypedDict, List

class TextToTokenize(TypedDict):
    title: str
    text: str

TEXTS_TO_TOKENIZE: List[TextToTokenize] = [{ "title": "Pan Tadeusz", "text": Path("../korpus-wolnelektury/pan-tadeusz-ksiega-1.txt").read_text()}, { "title": "The Pickwick Papers", "text": Path("../korpus-mini/the-pickwick-papers-gutenberg.txt").read_text()}, { "title": "Fryderyk Chopin" ,"text": Path("../korpus-mini/fryderyk-chopin-wikipedia.txt").read_text()}]
TOKENIZER_PATH = "tokenizers/tokenizer-all-corpora.json"

def visualize_tokens_with_gaps(text: str, encoding):
    tokens = encoding.tokens
    offsets = encoding.offsets

    print("\n" + "="*50)
    print(f"Oryginalny Tekst: '{text}'")
    print("="*50)

    visualized_sequence = []
    last_end_index = 0

    for i in range(len(tokens)):
        token = tokens[i]
        start, end = offsets[i]

        if start > last_end_index:
            gap = text[last_end_index:start]
            visualized_sequence.append(f"[GAP:'{gap}']")

        display_token = token
        if token.startswith(' '):
            display_token = f"TOKEN_BPE:'{token.lstrip(' ')}'"
        elif token.startswith('##'):
            display_token = f"TOKEN_WP_CONT:'{token.lstrip('##')}'"
        else:
             display_token = f"TOKEN:'{token}'"

        visualized_sequence.append(display_token)

        last_end_index = end

    if last_end_index < len(text):
         visualized_sequence.append(f"[GAP_END:'{text[last_end_index:]}']")

    print("Wizualizacja (ciąg tokenów i luk):")
    print(" ".join(visualized_sequence))
    print("="*50)

def main():
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Błąd: Plik tokenizera nie został znaleziony pod ścieżką: {TOKENIZER_PATH}")
        print("Upewnij się, że plik 'bielik-v3-tokenizer.json' znajduje się w tym samym katalogu.")
        return

    try:
        # Ładowanie istniejącego tokenizera z pliku JSON
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        print(f"Pomyślnie załadowano tokenizer z: {TOKENIZER_PATH}")
    except Exception as e:
        print(f"Błąd podczas ładowania tokenizera: {e}")
        return

    # Tokenizacja tekstów

    print(f"\nTokenizacja tekstów z wykorzystaniem tokenizera '{TOKENIZER_PATH.split('/')[-1]}':")
    for text in TEXTS_TO_TOKENIZE:
        encoding = tokenizer.encode(text["text"])
        # Wstępne wypisanie wyników
        print(f"\nWyniki Tokenizacji dla {text['title']}:")
        print(f"Tokeny: {len(encoding.tokens)}")
        # print(f"Offsets: {len(encoding.offsets)}")


    # Użycie funkcji wizualizującej
    # visualize_tokens_with_gaps(TEXT_TO_TOKENIZE, encoding)

if __name__ == "__main__":
    main()