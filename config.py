"""Settings shared by several pipeline steps."""

EMBED_MODEL = "intfloat/multilingual-e5-small"   # CPU-friendly, German + English
EMBED_MAX_TOKENS = 512                           # longer inputs are truncated by the model
INDEX_DIR = "index"                              # default location of the chunk index

# Generation LLM. Each value can be overridden by an environment variable of the same
# name; LLM_API_KEY holds the key. See llm.py for the supported providers.
LLM_PROVIDER = "gemini"
LLM_MODEL = None                                 # None: the default model of the provider
LLM_URL = None                                   # None: the default endpoint of the provider
