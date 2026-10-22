# PCCOE College Knowledge Assistant — Research Paper Context Pack

Use this document as the project brief when asking another model to help draft a research paper. It describes the implementation visible in this repository and labels historical measurements and unresolved limitations. Do not present unmeasured claims as results.

## 1. Project identity

**Working title:** PCCOE College Knowledge Assistant: A Source-Grounded, Multilingual Retrieval-Augmented System for Institutional Information

**Project type:** Student capstone / applied software and information-retrieval project.

**Institutional scope:** Pimpri Chinchwad College of Engineering (PCCOE), Pune, and its official `pccoepune.com` domain family. This is a domain-scoped college knowledge assistant, not an unrestricted web search engine and not a model trained/fine-tuned on PCCOE data.

**Problem:** Students ask varied questions in natural language (including English, Hindi, Marathi, and Romanized Hindi/Marathi) about fees, syllabus, faculty, HOD contacts, placements, clubs, exams, notices, admissions, policies, and services. Relevant facts may be buried in linked department pages, tables, PDFs, and other office documents. A plain vector search can return adjacent-but-wrong pages, fragments, navigation text, or stale/conflicting sources. The project aims to retrieve source passages, produce a readable answer grounded in them, and show concise links/page references so users can verify the answer.

**Intended users:** PCCOE students and prospective students seeking institutional information. Staff may also use it for quick lookup, but the current system is not an official administrative authority.

## 2. Product requirements established during development

- Keep the existing report-inspired PCCOE interface and provide a ready-to-run local web application.
- Accept open-ended questions instead of relying on a predefined question/answer list.
- Crawl PCCOE pages and their internal links, including department websites, and index linked source documents where permitted.
- Parse document content (especially PDFs and tables) so answers contain information, not only lists of links.
- Support syllabus, fees, faculty/HOD, placements, clubs, schedules, notices, policy, and other student-information topics without hard-coded answers for each question.
- Return concise readable responses with only relevant official references; PDF citations should include page anchors where known.
- Reduce unsupported answers and false matches. Ask a short clarification for genuinely underspecified questions, preserve conversation context, and support multilingual questions/replies when a model provider is configured.
- Provide an “Update Info”/sync capability and persistent indexing so data survives restarts and can be refreshed.
- Make coverage observable so partial crawl status is not represented as complete PCCOE coverage.

## 3. Architecture and technology

### Front end

- React + TypeScript, bundled with Vite.
- Existing PCCOE report-style interface is retained in `src/App.tsx` and styles in `src/style.css` / `src/onboarding.css`.
- Chat sessions, references, loading/status states, prompt suggestions, and the update/coverage experience are presented through the web UI.

### API / orchestration

- Python 3.11+ with FastAPI and Uvicorn (`app/main.py`).
- API provides health, direct query, persistent chat, file ingestion, source inventory, scraper status/sync/history, coverage and statistics endpoints.
- APScheduler enables periodic recrawling (default 24 hours; configurable or disabled).
- Conversation messages are stored in SQLite (`app/conversations.py`), capped to a bounded recent history per session. The retrieval/answer flow receives recent conversation context to resolve follow-ups.

### Indexes

- ChromaDB persistent vector collection (`pccoe_sources`) for semantic retrieval.
- SQLite FTS5 index alongside Chroma for exact lexical matches such as names, course codes, dates, and amounts (`app/store.py`). FTS may reconcile itself from Chroma records if the indexes diverge.
- Content is split into overlapping paragraph/sentence chunks. Metadata stores URL/source identity, title, chunk number/hash, page, category, inferred department, academic year, document type, and sync timestamps.
- Updated chunks are upserted; unchanged chunks are retained; stale chunks are pruned after successful document replacement.

### Query and answer flow

1. Preserve recent chat history and analyze the latest query (`app/query_understanding.py`). Extract topic, department, program/year, and requested information; ask a follow-up when broad scope makes an answer ambiguous.
2. Build semantic and lexical searches (`app/agent.py`, `app/store.py`). Use intent expansions for common broad areas (e.g., faculty, placements, syllabus, fees, clubs), multilingual fallback mapping, and optional LLM-generated query rewrites. Query rewrites are search hints, not evidence.
3. Fuse dense Chroma and FTS results, with metadata/category/department/year filtering and relevance checks.
4. Expand a promising source page/document when the question needs a list, roster, or facts likely split across chunks. Apply specialized evidence formatting for some structured topics (faculty, HOD/contact, clubs, placements, fees, syllabus and schedules) while using a general grounded-answer path for other questions.
5. Compose an answer from retrieved evidence through Gemini, Groq, or Ollama when configured. Otherwise use an extractive answerer. PCCOE-specific claims are instructed to cite evidence; unsupported answers should abstain rather than fill gaps from model memory.
6. Return a concise set of relevant source links, with PDF page anchors where available. Source links remain separately inspectable in the UI.

**Important distinction:** This is retrieval-augmented generation (RAG), not “training” the foundation model on PCCOE. The local index stores source text and embeddings. The model plans queries and/or writes a grounded answer at request time; it should not be treated as the authority for current institutional facts.

## 4. Crawling and document ingestion

### Crawler (`app/crawler.py`)

- Begins from the main PCCOE site plus configured department and high-value resource seeds (`app/config.py` / `.env.example`).
- Follows internal links and sitemaps within the configured allowed domain family; checks URLs to reduce SSRF risk and follows `robots.txt` rules.
- Has a configurable per-run page cap (default 5,000 new pages), a persistent queue/checkpoint, status reporting, and restart continuation.
- Records processed/failed/skipped counts and per-host coverage. `GET /api/coverage` exposes inventory/coverage information; `GET /api/scraper/status` exposes crawl progress.
- Supports HTML and common linked documents: PDF, DOCX, PPTX, XLSX, CSV; text/Markdown uploads are also supported by the ingestion flow.
- Per-page download limit in current code is 40 MB; long requests/timeouts, blocked hosts, unavailable URLs and unsupported file types can still prevent ingestion.
- It does not crawl arbitrary unrelated domains. Robots exclusions are intentional and should not be bypassed. An administrator may need to obtain permission or an official export when a host blocks crawling.

### Parsing (`app/ingest.py`)

- PDF: extract text page by page and record page metadata; preserve PDF page anchors in citations.
- DOCX: extract paragraphs and table rows.
- PPTX: extract slide text and tables.
- XLSX: extract sheet names and row values.
- CSV/TXT/Markdown: decode and ingest text.
- Optional Gemini OCR can attempt transcription of low-text/image PDFs if a valid Gemini API key is configured. OCR failures are swallowed to preserve crawl continuity; OCR output is not guaranteed correct and should be validated for critical figures/names.
- HTML parsing preserves semantic headings, lists, tables and some profile-card-like blocks, while attempting to omit repeated navigation/footer content.

## 5. Configuration / run commands

Copy `.env.example` to `.env`, then set keys locally. Never commit `.env` or include live secret values in a paper or prompt.

- `LLM_PROVIDER=auto|gemini|groq|ollama|mock`
- `GEMINI_API_KEY`, `GROQ_API_KEY` (optional; server-side only)
- Model defaults in current config: Gemini `gemini-3.8-flash`, Groq `llama-3.3-70b-versatile` (verify model availability before deployment)
- `CHROMA_PATH`, `DATA_PATH`, `COLLEGE_URL`, `ALLOWED_DOMAINS`, `SEED_URLS`
- `ADMIN_TOKEN` protects update/scraper admin routes; it must be set to a private random value before enabling administrative sync.
- `CRAWL_INTERVAL_HOURS` defaults to 24; `MAX_CRAWL_PAGES` defaults to 5,000.

Typical local run:

```powershell
pip install -r requirements.txt
npm install
Copy-Item .env.example .env
# add API key and private ADMIN_TOKEN to .env if desired
uvicorn app.main:app --reload --port 8000
# second terminal
npm run dev
```

The frontend Vite URL is shown in its terminal (often `http://localhost:5173`); API docs are at `http://127.0.0.1:8000/docs`.

## 6. Source/site scope

Configured seeds and discovered links cover the main college site and department subdomains including Computer Engineering, IT, ENTC, Computer Engineering regional language, Civil, Mechanical, AI/ML, First Year Engineering, MCA, B.Voc, and some PCCOE research/initiative sites where linked. Seeded content includes fee structures, academic calendar, exam timetable, notices, department syllabi, selected faculty pages, and other high-value PDFs.

This is not proof that every PCCOE page is present. Coverage depends on discoverable links/sitemaps, seed completeness, crawl limits, DNS/availability, format support, parser quality, robots policy, and successful document extraction. The Mechanical host was previously observed returning a robots policy that disallowed crawling; recheck current policy before describing this as current. Duplicate URLs/aliases may also occur.

## 7. Current evidence and status in repository (dated snapshots; verify again before publication)

### Historical retrieval evaluation

`data/evaluation_report.json` is dated **2026-10-16** and reports:

- 104 test questions across 26 student-information topics.
- Languages: English, Hindi, Romanized Hindi / Hinglish, Marathi (four versions per topic in the documented audit design).
- At that snapshot, 15,395 chunks were in the index.
- 80 / 104 queries (77%) met the evaluator’s retrieval/source-coverage proxy.
- By language, README reports English 21/26, Hindi 20/26, Hinglish 20/26, and Marathi 19/26.
- Historical baseline comparison described in report: 47/104 legacy pattern hits. Treat this as an internal baseline only; verify exact evaluator definitions before claiming a comparative improvement.
- The evaluator measures retrieved passage/topic/source coverage and citations, not semantic answer correctness, factual precision, hallucination rate, user satisfaction, or generation quality. It is not a human-reviewed benchmark.

### Historical crawl snapshot

The last saved `data/crawl_status.json` inspected for this context showed a **partial** run: 4,076 processed, 582 failed, 358 skipped, 5,000 discovered, and 501 queued. It listed multiple department/initiative hosts. This is a snapshot from a previous crawl, not a live status guarantee. The index count and crawl state may have changed since. Failures included network/DNS errors for event/gallery assets; crawling can be partial even when much of the main site is indexed.

### Historical local verification

The README claims that, at its last update, 12 backend tests passed, frontend production build succeeded, and representative live questions returned grounded answers for Mechanical HOD, Computer placements, engineering fees, and Research Methodology. This document has not rerun those checks. Do not imply they were run on the current checkout unless rerun and timestamped.

## 8. Limitations and risks to state clearly

- Coverage is incomplete and crawl status can be partial. “All PCCOE knowledge” is an aspiration, not a demonstrated property.
- `robots.txt`, server availability, crawling limits, dynamic JavaScript pages, authentication, file-size/timeouts, DNS, and unsupported formats restrict discovery/extraction.
- PDF text extraction and OCR can lose table structure, reading order, names, or numeric values. Scanned documents without configured OCR may be unsearchable.
- Search relevance and answer correctness are distinct. A relevant-looking reference does not prove the generated answer is correct. Citation-to-claim support needs a separate audit.
- The optional LLM provider status in the README snapshot was `mock`; if no API key is configured, multilingual query understanding and fluent answer generation are limited to fallback/extractive behavior. Verify current `.env` locally without revealing secrets.
- Query evaluation is a small internally authored benchmark (104 questions); it may not represent real user traffic, all PCCOE branches, all languages, ambiguous questions, or multi-hop questions. It lacks a human-annotated gold answer and statistical uncertainty analysis.
- No evidence in this repository establishes production load, latency under concurrency, security penetration testing, accessibility, user study, or deployment uptime. Do not claim these without new evidence.
- Freshness of facts such as staff roles, fees, schedules and notices must be attributed to the source’s publication date/academic year and crawl time.
- API credentials pasted into chat previously should be revoked/rotated; never reproduce them in research materials.

## 9. Research questions / contribution candidates (frame as candidates, not established contributions)

1. Does combining semantic vector retrieval with lexical FTS improve retrieval coverage for entity-heavy college queries (names, course codes, dates, amounts) over vector-only retrieval?
2. Does page/document expansion improve completeness for roster, faculty, syllabus and table questions without increasing irrelevant citations?
3. How much does query rewriting / multilingual normalization affect retrieval across English, Hindi, Hinglish and Marathi?
4. How do parsing method and document format (HTML, digital PDF, scanned PDF, DOCX, XLSX) affect evidence recall and answer faithfulness?
5. What are the trade-offs between crawl coverage, freshness, crawl cost, latency and answer quality?

To answer these experimentally, define a gold test set, query-level source/evidence labels, answer correctness and faithfulness scoring (preferably two human annotators), a leakage-safe split, baselines (keyword-only, vector-only, hybrid), ablations, latency/cost measurement, and reproducible run timestamps. The existing 104-query report alone is not enough to answer these questions rigorously.

## 10. Suggested paper outline

1. Abstract (write after results are finalized)
2. Introduction and motivation
3. Problem definition, scope and requirements
4. Related work (retrieve and cite verified primary/peer-reviewed literature; do not invent citations)
5. System architecture and implementation
6. PCCOE source discovery, crawling, parsing and index update strategy
7. Query understanding, hybrid retrieval, source expansion, grounded generation and citation UX
8. Experimental setup (dataset construction, languages, topics, ground truth, baselines, metrics, software/model versions)
9. Results (only measured results; distinguish retrieval from answer quality)
10. Error analysis and case studies (wrong department, stale source, irrelevant snippets, incomplete table/list, OCR failure, ambiguity)
11. Limitations, privacy, robots/website policy and responsible operation
12. Conclusion and future work

## 11. Claude-ready instruction

You are helping write an academic research paper about the PCCOE College Knowledge Assistant described above. First analyze the project brief and the repository artifacts supplied with it. Draft a technically accurate paper in a conventional academic structure. Do not invent experiments, user studies, deployments, sources, numbers, model capabilities, citations, or claims of complete PCCOE coverage. Clearly label the 2026-10-16 evaluation/crawl figures as historical internal measurements, and explain that 80/104 is retrieval/source-coverage proxy, not answer accuracy. Separate implemented features from proposed future work. If a required fact or result is missing, insert a clearly marked `[AUTHOR TO VERIFY/PROVIDE: ...]` placeholder and list exactly what evidence is needed. Use real peer-reviewed or primary sources for related work, verify every bibliographic item, and use a consistent citation style. Never include API keys or secrets. Before writing the final paper, provide a claim-evidence table mapping each quantitative/technical claim to a repository file or experiment artifact. Avoid claiming this system works like ChatGPT/Google in general; accurately describe it as a PCCOE-domain-scoped RAG assistant.

## 12. Repository files to inspect/cite for implementation claims

- `README.md` — setup, architecture summary, scope and historical audit notes.
- `app/main.py` — FastAPI orchestration, answer generation/formatting, session endpoints, ingestion/admin/status/coverage endpoints.
- `app/agent.py` — multilingual query variants, retrieval query planning, source expansion.
- `app/store.py` — Chroma and SQLite FTS indexing, chunking, hybrid retrieval.
- `app/crawler.py` — safe URL handling, robots-aware crawling, crawl queue, status/checkpoint and document discovery.
- `app/ingest.py` — PDF/DOCX/PPTX/XLSX/CSV/text parsing and optional OCR.
- `app/query_understanding.py` — intent/entities and clarification logic.
- `app/taxonomy.py` — department aliases, topic categories, academic-year inference.
- `app/conversations.py` — persistent SQLite chat history.
- `app/config.py` and `.env.example` — runtime settings and seeds (do not include secrets).
- `app/evaluate.py`, `data/question_bank.json`, `data/evaluation_report.json` — evaluation procedure, questions and historical retrieval-only report. Inspect evaluator definitions before quoting a metric.
- `tests/test_knowledge.py` — backend test cases (test count/status must be rerun before publication).
- `src/App.tsx`, `src/style.css`, `src/onboarding.css` — frontend behavior and design.
- `requirements.txt`, `package.json` — software dependencies.

---

**Prepared from repository inspection:** 2026-10-22 (Asia/Kolkata). Historical measurements carry their own earlier timestamps. Recheck code, index, crawler status, model provider, and tests before submitting a paper.
