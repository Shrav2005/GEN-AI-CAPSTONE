# Environment configuration

This guide describes the variables used by the PCCOE assistant. Keep this file
and `.env.example` in the project root, alongside your private `.env` file.

## Set up `.env`

In the project root, copy `.env.example` to `.env`, then edit `.env` locally:

```powershell
Copy-Item .env.example .env
```

The `.env` file is ignored by Git. Do not paste API keys, admin tokens, or other
private values into this guide, `.env.example`, source files, screenshots, or
Git commits. Use the placeholders below only as a reference; replace them in
your local `.env`. If a credential was exposed, revoke it with its provider and
create a replacement.

## Variables

| Variable | Purpose |
| --- | --- |
| `LLM_PROVIDER` | Provider selection: `auto`, `gemini`, `groq`, `ollama`, or `mock`. `auto` selects a configured Gemini or Groq provider, otherwise the local extractive answerer. |
| `GEMINI_API_KEY` | Private Gemini API key. Leave empty if unused. |
| `GROQ_API_KEY` | Private Groq API key. Leave empty if unused. |
| `GEMINI_MODEL` | Gemini model name. Defaults to `gemini-3.8-flash`. |
| `GROQ_MODEL` | Groq text-generation model. Defaults to `openai/gpt-oss-120b`. |
| `GROQ_OCR_MODEL` | Groq vision model used as an OCR fallback. Defaults to `qwen/qwen3.8-27b`. |
| `OLLAMA_BASE_URL` | Base URL for a local Ollama service. Defaults to `http://localhost:11434`. |
| `OLLAMA_MODEL` | Local Ollama model name. Defaults to `llama3.2`. |
| `CHROMA_PATH` | Persistent vector-index directory. Defaults to `./data/chroma`. |
| `DATA_PATH` | Application data directory. Defaults to `./data`. |
| `COLLEGE_URL` | Main PCCOE website URL. Defaults to `https://www.pccoepune.com/`. |
| `ALLOWED_DOMAINS` | Comma-separated domains the crawler may access. Defaults to `pccoepune.com`. |
| `SEED_URLS` | Optional comma-separated official PCCOE URLs to seed the crawler. If you customize it, include the desired start pages; the application also supplies its core admission seeds. |
| `ADMIN_TOKEN` | Private token required by protected crawl/update routes. Set a long, random value; leave empty to keep those routes disabled. |
| `CONFIDENCE_THRESHOLD` | Retrieval confidence threshold. Defaults to `0.50`. |
| `LIVE_WEB_FALLBACK` | Whether the assistant may try live official-site discovery when indexed evidence is insufficient. Defaults to `true`. |
| `MAX_CRAWL_PAGES` | Maximum new pages per crawl batch. Defaults to `5000`. |
| `CRAWL_INTERVAL_HOURS` | Scheduled crawl interval in hours. Defaults to `24`; set to `0` to disable scheduled crawls. |
| `FRONTEND_ORIGIN` | Frontend origin allowed by the API. Defaults to `http://localhost:5173`. |

## Safe example

Use this as a reference, not as a replacement for the project's `.env.example`.
Do not put real credentials in this document:

```dotenv
LLM_PROVIDER=auto
GEMINI_API_KEY=
GROQ_API_KEY=
GEMINI_MODEL=gemini-3.8-flash
GROQ_MODEL=openai/gpt-oss-120b
GROQ_OCR_MODEL=qwen/qwen3.8-27b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
CHROMA_PATH=./data/chroma
DATA_PATH=./data
COLLEGE_URL=https://www.pccoepune.com/
ALLOWED_DOMAINS=pccoepune.com
# Optional: set SEED_URLS to comma-separated official PCCOE URLs.
ADMIN_TOKEN=
CONFIDENCE_THRESHOLD=0.50
LIVE_WEB_FALLBACK=true
MAX_CRAWL_PAGES=5000
CRAWL_INTERVAL_HOURS=24
FRONTEND_ORIGIN=http://localhost:5173
```

The full default seed list is maintained in `.env.example` and application
configuration. To run the API and frontend, follow the setup instructions in
[`README.md`](./README.md).
