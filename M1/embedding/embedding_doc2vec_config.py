# import z corpora (zakładam, że jest to plik pomocniczy)
from corpora import CORPORA_FILES # type: ignore

# files = CORPORA_FILES["ALL"]
files = CORPORA_FILES["WOLNELEKTURY"]
# files = CORPORA_FILES["PAN_TADEUSZ"]

TOKENIZER_FILE = "../tokenizer/tokenizers/bielik-v3-tokenizer.json"
# TOKENIZER_FILE = "../tokenizer/tokenizers/tokenizer-all-corpora-128K.json"
OUTPUT_MODEL_FILE = "doc2vec_model_combined.model"
OUTPUT_SENTENCE_MAP = "doc2vec_model_sentence_map_combined.json"

# Parametry treningu Doc2Vec
VECTOR_LENGTH = 20
WINDOW_SIZE = 6
MIN_COUNT = 2
WORKERS = 8
EPOCHS = 20
SG_MODE = 0
