# import z corpora (zakładam, że jest to plik pomocniczy)
from corpora import CORPORA_FILES # type: ignore

# --- KONFIGURACJA ŚCIEŻEK I PARAMETRÓW ---
# files = CORPORA_FILES["WOLNELEKTURY"]
# files = CORPORA_FILES["PAN_TADEUSZ"]
files = CORPORA_FILES["ALL"]

TOKENIZER_FILE = "../tokenizer/tokenizers/tokenizer-all-corpora-16K.json"
# TOKENIZER_FILE = "../tokenizer/tokenizers/bielik-v1-tokenizer.json"
# TOKENIZER_FILE = "../tokenizer/tokenizers/bielik-v3-tokenizer.json"

OUTPUT_TENSOR_FILE = "embedding_tensor_cbow_16K.npy"
OUTPUT_MAP_FILE = "embedding_token_to_index_map_16K.json"
OUTPUT_MODEL_FILE = "embedding_word2vec_cbow_model_16K.model"

# Parametry treningu Word2Vec (CBOW)
VECTOR_LENGTH = 64
WINDOW_SIZE = 6
MIN_COUNT = 2
WORKERS = 8
EPOCHS = 20
SAMPLE_RATE = 1e-2
SG_MODE = 0 # 0 dla CBOW, 1 dla Skip-gram