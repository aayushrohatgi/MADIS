# LangGraph Content Extraction Agent

A multi-agent workflow built with [LangGraph](https://github.com/langchain-ai/langgraph) and Google Gemini that extracts mobile numbers and URLs from hosted files. 
Supports document and image files, parallel extraction, automatic retries, Postgres state checkpointing, and error replay/forking.
As this is just a POC its missing some much-needed guard rails like input validations, file safety and file size constraints, rate limits etc.
---

## Features

- Extracts **URLs** and **phone numbers** (with country code inference) from any supported file type
- **Image support** — OCR + automatic translation to English via Gemini Flash before extraction
- **Parallel execution** — URL and phone pipelines run simultaneously
- **Automatic retries** — all tool calls retry up to 2 times with exponential backoff
- **Postgres checkpointing** — full graph state saved after every node via LangGraph's built-in checkpointer
- **Error checkpointing** — failed runs saved to Postgres with node name, error, and state snapshot
- **Replay & fork** — resume failed runs or branch them with corrected state

---

## Workflow

```
START
  └─→ Agent 1: File Processor
        ├─ download_file(file_url)
        ├─ check_extension(bytes)      ← detects extension + whether it's an image
        ├─ is_extension_supported(ext)
        └─ extract_content(bytes, ext) ← skipped for image formats
              │
              ├─→ unsupported / error ──────────────────────────────────→ END
              │
              ├─→ image format
              │     └─→ Node 1.5: OCR + Translation (Gemini Flash)
              │               - OCR: extract text from image bytes
              │               - Translate to English if not already English
              │               - Output: raw_text
              │                     │
              │                     ▼
              └─→ non-image ────────┤
                                    │
                          ┌─────────┴─────────┐  (parallel fan-out)
                          │                   │
                     Agent 2              Agent 4
                     URL Extractor        Phone Extractor
                     (Gemini Pro)         (Gemini Pro + country inference)
                          │                   │
                     Agent 3              Agent 5
                     URL Normalizer       Phone Normalizer
                     (tool call)          (tool call)
                          │                   │
                          └─────────┬─────────┘
                               Agent 6: Aggregator
                                    │
                                   END
                       { urls: [...], phone_numbers: [...] }
```

---

## Project Structure

```
langgraph-agent/
├── main.py                      # Entry point — run_agent(file_url)
├── config.py                    # Env config (Gemini, Postgres, Tools API)
├── requirements.txt
├── .env.example
├── replay_example.py            # CLI tool for replay and fork operations
│
├── graph/
│   ├── state.py                 # AgentState TypedDict
│   ├── graph.py                 # Graph definition, edges, Postgres checkpointer
│   └── nodes/
│       ├── file_processor.py    # Agent 1 — download, detect, extract (routes images to OCR)
│       ├── ocr_translation.py   # Node 1.5 — OCR + English translation (Gemini Flash)
│       ├── url_extractor.py     # Agent 2 — LLM URL extraction
│       ├── url_normalizer.py    # Agent 3 — URL normalization tool call
│       ├── phone_extractor.py   # Agent 4 — LLM phone extraction + country inference
│       ├── phone_normalizer.py  # Agent 5 — phone normalization tool call
│       └── aggregator.py        # Merges parallel branch results
│
├── tools/
│   ├── file_tools.py            # download_file, check_extension, is_supported, extract_content
│   └── normalizer_tools.py      # normalize_urls, normalize_phones
│
└── db/
    ├── checkpoint.py            # Custom error checkpoint save/load (Postgres)
    └── replay.py                # Replay and fork logic
```

---

## Prerequisites

- Python 3.11+
- Google Gemini API key
- PostgreSQL instance (local or cloud)
- Tools REST API (stubbed — replace base URL when ready)

---

## Setup

**1. Clone and create a virtual environment**
```bash
git clone <repo-url>
cd langgraph-agent
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure environment**
```bash
cp .env.example .env
```

Edit `.env` with your values:
```env
GEMINI_API_KEY=your_gemini_api_key_here

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agent_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password_here

TOOLS_API_BASE_URL=http://localhost:8000
```

**4. Start Postgres** (if running locally via Docker)
```bash
docker run --name langgraph-pg \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=agent_db \
  -p 5432:5432 \
  -d postgres:16
```

Postgres checkpoint tables are created automatically on first run.

---

## Usage

**Run the agent:**
```bash
python main.py "https://your-bucket.s3.amazonaws.com/yourfile.pdf"
```

Works for both document and image files:
```bash
# Document
python main.py "https://your-bucket.s3.amazonaws.com/document.pdf"

# Image (OCR + translation applied automatically)
python main.py "https://your-bucket.s3.amazonaws.com/scanned-form.jpg"
```

**Example output:**
```json
{
  "run_id": "3f2a1b4c-...",
  "status": "success",
  "urls": [
    "https://example.com",
    "https://another-site.com/path"
  ],
  "phone_numbers": [
    "+911234567890",
    "+14155551234"
  ]
}
```

**Possible status values:**

| Status | Meaning |
|---|---|
| `success` | Extraction completed successfully |
| `unsupported_file_type` | File format not supported by the tools API |
| `error` | A tool or LLM call failed after retries |

---

## Supported File Types

File type support is determined by the Tools API (`/files/is-supported`). The `check_extension` endpoint also returns an `is_image` flag which drives the OCR routing decision.

**Image formats** (routed through OCR + Translation):
`jpg`, `jpeg`, `png`, `gif`, `webp`

**Document formats** (routed through direct text extraction):
Any format supported by the Tools API (e.g. `pdf`, `docx`, `txt`, `html`)

---

## Image Processing (OCR + Translation)

When an image file is detected, the workflow automatically:

1. Encodes the image bytes and sends them to **Gemini Flash** (`gemini-3-flash-preview`)
2. Extracts all visible text from the image (OCR)
3. If the text is not in English, translates it to English
4. If the text is already in English, returns it unchanged
5. The resulting `raw_text` flows into the same URL and phone extraction pipeline as document files

Gemini Flash is used here (instead of Pro) because OCR + translation is a well-defined task that doesn't require the full reasoning capability of Pro — making it faster and cheaper.

---

## Replay & Fork

When a run fails, the error and state are saved to Postgres. You can then replay or fork the run without starting from scratch.

**List all failed runs:**
```bash
python replay_example.py list
```

**Inspect state at point of failure:**
```bash
python replay_example.py inspect <run_id>
```

**Replay** — resume the same run from last checkpoint (no state changes):
```bash
python replay_example.py replay <run_id>
```
Best for transient failures like network timeouts where the state is correct.

**Fork with a new URL** — re-runs the full workflow on a different file:
```bash
python replay_example.py fork-url <run_id> https://new-bucket.s3.amazonaws.com/fixed.pdf
```

**Fork with corrected raw text** — skips file processing, re-runs extraction onwards:
```bash
python replay_example.py fork-text <run_id> "corrected text content..."
```

**Fork with manual state overrides** — inject corrected extracted data, re-runs normalization only:
```bash
python replay_example.py fork-override <run_id>
```

### Replay vs Fork — When to Use Which

| Scenario | Use |
|---|---|
| Network timeout, API was down | `replay` |
| Wrong file was provided | `fork-url` |
| File extraction or OCR produced bad text | `fork-text` |
| LLM extracted wrong phones/URLs | `fork-override` |

---

## Postgres Tables

LangGraph auto-creates these tables on first run:

| Table | Purpose |
|---|---|
| `checkpoints` | Full graph state snapshot after every node |
| `checkpoint_blobs` | Binary state data (file bytes etc.) |
| `checkpoint_writes` | Pending writes between nodes |

Our custom error table:

| Table | Purpose |
|---|---|
| `error_checkpoints` | Failed node name, error message, state snapshot, timestamp |

**Query your runs:**
```sql
-- All runs
SELECT DISTINCT thread_id, created_at FROM checkpoints ORDER BY created_at DESC;

-- All checkpoints for a specific run (in execution order)
SELECT checkpoint_id, created_at, metadata->>'step' AS step
FROM checkpoints
WHERE thread_id = '<run_id>'
ORDER BY (metadata->>'step')::int ASC;

-- All failed runs
SELECT run_id, failed_node, error, created_at FROM error_checkpoints ORDER BY created_at DESC;
```

---

## Tools API Contract

All tools in `tools/` are stubbed as REST calls. Update `TOOLS_API_BASE_URL` in `.env` when your backend is ready. Expected request/response shapes:

| Endpoint | Method | Request                          | Response |
|---|---|----------------------------------|---|
| `/files/download` | POST | `{ "file_url": str }`            | raw file bytes |
| `/files/check-extension` | POST | `file` (multipart)               | `{ "extension": str, "is_image": bool }` |
| `/files/is-supported` | GET | `?extension=pdf`                 | `{ "supported": bool }` |
| `/files/extract-content` | POST | `file` + `extension` (multipart) | `{ "text": str }` |
| `/normalize/urls` | POST | `{ "urls": [str] }`              | `{ "normalized_urls": [str] }` |
| `/normalize/phones` | POST | `{ "phones": [str] }`            | `{ "normalized_phones": [str] }` |

---

## LLM Models Used

| Node | Model | Purpose |
|---|---|---|
| OCR + Translation | `gemini-3-flash-preview` | Fast, cheap — well-defined OCR/translation task |
| URL Extractor | `gemini-1.5-pro` | Reasoning over unstructured text |
| Phone Extractor | `gemini-1.5-pro` | Reasoning + country code inference from context |

---

## Configuration Reference

| Variable | Description | Default |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API key | required |
| `GEMINI_MODEL` | Gemini model for extraction nodes | `gemini-1.5-pro` |
| `POSTGRES_HOST` | Postgres host | `localhost` |
| `POSTGRES_PORT` | Postgres port | `5432` |
| `POSTGRES_DB` | Database name | `agent_db` |
| `POSTGRES_USER` | Database user | `postgres` |
| `POSTGRES_PASSWORD` | Database password | required |
| `TOOLS_API_BASE_URL` | Base URL for file/normalizer tools API | `http://localhost:8000` |
| `MAX_RETRIES` | Max retries for tool calls | `2` |

---

## Future Improvements

- Input validation on file URL before invoking the graph
- File size guardrail at `file_processor` level — reject oversized files early
- LLM output validation + retry loop for malformed JSON in extractor nodes
- PII masking in logs and Postgres checkpoints
- Overlapping chunk strategy for large documents within the accepted size limit