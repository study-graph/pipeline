"""Settings shared by several pipeline steps."""

EMBED_MODEL = "intfloat/multilingual-e5-small"   # CPU-friendly, German + English
EMBED_MAX_TOKENS = 512                           # longer inputs are truncated by the model
INDEX_DIR = "index"                              # default location of the chunk index
