"""Settings shared by several pipeline steps."""

EMBED_MODEL = "intfloat/multilingual-e5-small"   # CPU-friendly, German + English
EMBED_MAX_TOKENS = 512                           # longer inputs are truncated by the model
INDEX_DIR = "index"                              # default location of the chunk index

# Language models, one set of settings per role (see llm.py). Each value can be overridden
# by an environment variable of the same name; LLM_API_KEY and VERIFY_API_KEY hold the keys.
# Answer generation:
LLM_PROVIDER = "openai"
LLM_MODEL = "qwen3.6:35b-a3b-q4_K_M"
LLM_URL = "http://localhost:11434/v1/chat/completions"
LLM_REASONING = None                             # e.g. "none"; see llm.py
# Claim decomposition and verification; None falls back to the LLM_ setting.
VERIFY_PROVIDER = None
VERIFY_MODEL = None
VERIFY_URL = None
VERIFY_REASONING = None                          # e.g. "none"; see llm.py

# Calibration of answer confidence (see calibrate.py). It belongs to the verifier model
# and its probes, not to an index; without the file, confidence stays uncalibrated.
CALIBRATION_FILE = "calibration.json"
