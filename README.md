# Chat with your PDF — RAG app

A Streamlit app built from `_Langchain_pdf_.ipynb`: upload a PDF, ask
questions about it, get answers generated only from retrieved chunks
of that PDF (retrieval-augmented generation).

## How it works

```
PDF upload
   │
   ▼
PyPDFLoader            → extract text per page
   │
   ▼
RecursiveCharacterTextSplitter (500 chars, 50 overlap) → chunks
   │
   ▼
HuggingFaceEmbeddings (all-MiniLM-L6-v2) → vector per chunk
   │
   ▼
FAISS                  → in-memory vector index
   │
   ▼ (per question)
similarity_search(question, k=3) → top 3 relevant chunks
   │
   ▼
prompt = "Answer using this context: ... Question: ... Answer:"
   │
   ▼
Gemini 2.5 Flash (Google API) → generated answer
```

Embeddings and vector search still run entirely locally/free
(`sentence-transformers` + FAISS). Only the final answer-generation
step calls the Gemini API, so you need a Google API key for that part
only.

## Fixes vs. the original notebook

- **Pipeline task**: the notebook loaded `flan-t5-small` (a seq2seq
  model) with `pipeline("text-generation", ...)`. That task is for
  decoder-only models (like GPT-style models) — flan-t5 needs
  `"text2text-generation"`. Fixed in `app.py`.
- **Undefined variable**: the notebook's `ask_question()` function
  referenced `vectorstore`, which was never defined (the actual object
  was named `db`). Fixed by consistently using one variable.
- **No more Colab / hardcoded file**: replaced `google.colab.files.upload()`
  and the hardcoded `"cv1.pdf"` with a real Streamlit file uploader that
  works with any PDF, locally or deployed.
- **Caching**: embedding the PDF and loading the models happen once
  (`st.cache_resource`), not on every question — otherwise every
  question would re-read and re-embed the whole PDF from scratch.
- **Chat history + sources**: added a running chat view and an
  expandable "Sources used" section under each answer, showing exactly
  which chunks (and page numbers) the answer was grounded in — useful
  for spotting when the model is guessing vs. actually citing the PDF.
- **Upgraded to Gemini 2.5 Flash**: swapped the local `flan-t5-small`
  model out for `gemini-2.5-flash` via `langchain-google-genai`, since
  flan-t5-small gave shallow, often inaccurate answers. Embeddings and
  FAISS retrieval are unchanged and still run locally for free — only
  the final answer-generation call goes to the Gemini API.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Get a Google API key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
   and create a free API key.
2. Either set it as an environment variable before running the app:
   ```bash
   export GOOGLE_API_KEY="your-key-here"     # set GOOGLE_API_KEY=... on Windows
   ```
   or just paste it into the sidebar text box when the app opens — it
   only lives in that browser session, never written to disk.

Gemini 2.5 Flash has a generous free tier for API usage as of writing,
but rate limits and pricing can change — check the
[Gemini API pricing page](https://ai.google.dev/gemini-api/docs/pricing)
if you're planning heavier usage.

## Run

```bash
streamlit run app.py
```

Upload a PDF, wait for it to finish indexing, then ask questions in the
chat box at the bottom.

### Deploying with a saved key

If you deploy this (e.g. Streamlit Community Cloud), set `GOOGLE_API_KEY`
as a secret/environment variable on the host rather than typing it into
the sidebar each time — the app checks `os.environ["GOOGLE_API_KEY"]`
first and only falls back to the sidebar prompt if that's unset.

## Known limitations (worth knowing before you rely on this)

- **No OCR** — scanned PDFs (images of text, no embedded text layer)
  will index 0 chunks and the app will say so. Add an OCR step
  (`pytesseract`, or `unstructured` with OCR extras) if you need to
  handle those.
- **In-memory only** — the FAISS index lives in memory for the current
  session; re-uploading later or restarting the app re-indexes from
  scratch. For a persistent knowledge base across sessions, save the
  FAISS index to disk (`db.save_local(...)` / `FAISS.load_local(...)`)
  or swap in a persistent vector DB (Chroma, Pinecone, Weaviate, etc.).
- **Single PDF at a time** — this app indexes whatever PDF is currently
  uploaded. For multi-document Q&A, extend `build_vectorstore` to
  accept multiple files and merge their chunks into one index.
