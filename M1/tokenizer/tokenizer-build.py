from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from corpora import get_corpus_file
import argparse

parser = argparse.ArgumentParser(
    description="Build and train a custom BytePair tokenizer",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python tokenizer-build.py --vocab-size 512 --text "Hello world" --merges 50
  python tokenizer-build.py --input-file data.txt --vocab-size 256 --merges 100
  python tokenizer-build.py --help
        """
)

parser.add_argument(
    "--output-file-name",
    type=str,
    default="custom-tokenizer.json",
    help="Name of the saved tokenizer file (default: custom-tokenizer.json)"
)

parser.add_argument(
    "--corpus-name",
    type=str,
    default="ALL",
    help="Name of the corpus to use for training (default: ALL)"
)

args = parser.parse_args()

TOKENIZER_OUTPUT_FILE = f"tokenizers/{args.output_file_name}.json"

# 1. Initialize the Tokenizer (BPE model)
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 2. Set the pre-tokenizer (e.g., split on spaces)
tokenizer.pre_tokenizer = Whitespace()

# 3. Set the Trainer
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=128000,
    min_frequency=2
)

FILES = [str(f) for f in get_corpus_file(args.corpus_name, "*.txt")]
print(FILES)

# 4. Train the Tokenizer
tokenizer.train(FILES, trainer=trainer)

# 5. Save the vocabulary and tokenization rules
tokenizer.save(TOKENIZER_OUTPUT_FILE)

for txt in [
    "Litwo! Ojczyzno moja! ty jesteś jak zdrowie.",
    "Jakże mi wesoło!",
    "Jeśli wolisz mieć pełną kontrolę nad tym, które listy są łączone (a to jest bezpieczniejsze, gdy słownik może zawierać inne klucze), po prostu prześlij listę list do spłaszczenia.",
]:
    encoded = tokenizer.encode(txt)
    print("Zakodowany tekst:", encoded.tokens)
    print("ID tokenów:", encoded.ids)
