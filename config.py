"""Settings shared by several pipeline steps.

Values are read from environment variables. A .env file in the project root is loaded
automatically; variables already set in the shell take priority over .env.
"""
import os

from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")
EMBED_MAX_TOKENS = int(os.environ.get("EMBED_MAX_TOKENS", "512"))

INDEX_DIR = os.environ.get("INDEX_DIR", "index")

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.6:35b-a3b-q4_K_M")
LLM_URL = os.environ.get("LLM_URL", "http://localhost:11434/v1/chat/completions")
LLM_REASONING = os.environ.get("LLM_REASONING")

VERIFY_PROVIDER = os.environ.get("VERIFY_PROVIDER")
VERIFY_MODEL = os.environ.get("VERIFY_MODEL")
VERIFY_URL = os.environ.get("VERIFY_URL")
VERIFY_REASONING = os.environ.get("VERIFY_REASONING")

CALIBRATION_FILE = os.environ.get("CALIBRATION_FILE", "calibration.json")
