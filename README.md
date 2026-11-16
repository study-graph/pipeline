# RAG pipeline for lecture material

This repository turns lecture material (slide decks, scripts, papers) into a searchable
index and answers questions from it with cited sources.

The pipeline runs locally. The embedding model runs on the CPU, the index is a directory
of files, and the language model that writes the answers is configurable, including a
local one.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- For `ask`: access to a language model, either a local server such as Ollama or an API key

## Setup

```sh
uv sync
```

The first `ingest` downloads the embedding model `intfloat/multilingual-e5-small`
(about 470 MB) from Hugging Face. Afterwards the pipeline works offline; setting
`HF_HUB_OFFLINE=1` skips the update check and silences the related warning.

## Usage

### Add documents to the index

```sh
uv run python pipeline.py ingest slides.pdf script.md
uv run python pipeline.py ingest corpus/               # every supported file below corpus/
uv run python pipeline.py ingest slides.pdf --save-steps debug/
```

Supported inputs are PDF and plain text or markdown. The type is detected from the file
content, not from the extension. A file whose content has not changed since its last
ingest is skipped; of a changed file, only chunks with new content are embedded again.

| Option | Effect |
| --- | --- |
| `--index DIR` | Index directory, default `index/` |
| `--force` | Process unchanged files as well |
| `--save-steps DIR` | Also write the parsed and the cleaned text there, for inspection |

### Search without a language model

```sh
uv run python embedder.py "Wie funktioniert der 3-way Handshake?"
```

Prints the five most similar chunks with document, page range, heading path and
similarity score. Use this to judge retrieval on its own.

### Ask a question

```sh
uv run python pipeline.py ask "Wie funktioniert der 3-way Handshake?"
uv run python pipeline.py ask "..." --prompt-only      # print the prompt, call no model
```

The retrieved chunks are numbered `[1]`, `[2]`, ... in the prompt and the model is told
to cite these numbers. After generation, every cited number is checked against the
sources that were given. The output lists the cited sources with document, page and
chunk id.

| Exit code | Meaning |
| --- | --- |
| 0 | Answer produced; a warning is printed if it cites no source |
| 1 | Index empty, model unreachable, or configuration incomplete |
| 2 | The answer cites a source number that was not given |

The check establishes that a cited source exists. It does not establish that the source
supports the statement.

## Language model configuration

The provider is set in `config.py` or, with priority, through environment variables.

| Variable | Meaning | Default |
| --- | --- | --- |
| `LLM_PROVIDER` | `gemini` or `openai` | `gemini` |
| `LLM_MODEL` | Model name | `gemini-3.8-flash` for `gemini`; required for `openai` |
| `LLM_URL` | Endpoint | The provider's public endpoint |
| `LLM_API_KEY` | API key | Required for `gemini`, optional for `openai` |

`openai` stands for any endpoint that speaks the OpenAI chat completions format. This
includes local servers.

Local model with Ollama (run `ollama serve` in another terminal):

```sh
export LLM_PROVIDER=openai
export LLM_URL=http://localhost:11434/v1/chat/completions
export LLM_MODEL=gemma4:31b
```

## How it works

`ingest` chains five steps as function calls. Each step is a module that can also be run
on its own for debugging.

| Step | Module | What it does |
| --- | --- | --- |
| Parse | `parse.py` | Extracts text page by page. In PDFs, lines set in a larger font than the body text become markdown headings. |
| Clean | `clean.py` | Removes page chrome (headers, footers, navigation bars, page numbers), which it recognises as lines recurring on neighbouring pages. Turns each slide into a titled section and merges the steps of an animated slide. |
| Chunk | `chunker.py` | Splits sections into chunks of at most 480 tokens of the embedding model, with a 50-token overlap, and merges very small neighbouring slides. Each chunk carries document, heading path, page range and a content-hash id. |
| Embed | `embedder.py` | Encodes chunks with `multilingual-e5-small`. |
| Store | `store.py` | Writes chunks and vectors to the index. |

`ask` embeds the question with the same model, takes the five most similar chunks,
builds the prompt (`answer.py`) and calls the language model through `llm.py`.

Two modules isolate what is expected to change: `store.py` is the only code that knows
the index layout, and `llm.py` is the only code that knows a provider API.

### Index layout

```
index/
  chunks.json       one object per chunk: id, doc, path, pages, text, embed_text
  embeddings.npy    one row per chunk, in the same order
  meta.json         embedding model, vector dimension, content hash per document
```

An index is bound to one embedding model. Writing to it with another model is refused,
because vectors of different models are not comparable.

### Settings

Shared settings are in `config.py` (embedding model, index directory, language model).
Thresholds of a single step are constants at the top of its module, for example
`MAX_TOKENS` and `OVERLAP` in `chunker.py` or `MIN_RUN` in `clean.py`.

## Known limitations

- Scanned PDFs without a text layer yield no text; there is no OCR.
- PowerPoint files are not supported; export them to PDF.
- Titles are recognised by font size. A title in bold type of the same size as the body
  text is not recognised; the first line below the page header is used instead.
- A document is identified by its file name without the extension. Two files with the
  same name in different directories overwrite each other in the index.
- There is no command to remove a document from the index.
