import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from google.genai.errors import ServerError
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loads variables from a local .env file (if present) into the process
# environment, so GOOGLE_API_KEY only needs to be set once, in .env,
# instead of pasted into the sidebar every run. Safe to call even if
# .env doesn't exist — it's a no-op in that case.
load_dotenv()

# ============================================================
# Chat with your PDF — RAG app
# Based on _Langchain_pdf_.ipynb
#
# Pipeline: PDF -> chunks -> embeddings (local) -> FAISS (local)
#           -> retrieve -> Gemini 2.5 Flash answer
#
# Embeddings + vector search stay local/free. Only the final answer
# generation step calls the Gemini API, which needs a Google API key.
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-3.8-flash"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

st.set_page_config(page_title="Chat with your PDF", page_icon="📄", layout="centered")

st.title("📄 Chat with your PDF")
st.write(
    "Upload a PDF, then ask questions about it. Answers are generated "
    "only from chunks of the PDF retrieved as relevant context (RAG), "
    f"using {LLM_MODEL}."
)

# ------------------------------------------------------------
# API key — checked in this order:
#   1. GOOGLE_API_KEY in a local .env file (loaded above) — set once,
#      never re-entered. This is the recommended way to run locally.
#   2. GOOGLE_API_KEY already set in the shell/host environment (e.g.
#      a secret set on your deployment platform).
#   3. Sidebar text box, as a one-off fallback if neither is set.
# Never hardcode a key in the source. See README for .env setup.
# ------------------------------------------------------------
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    with st.sidebar:
        st.subheader("Google API key")
        st.caption(
            "Tip: create a `.env` file with `GOOGLE_API_KEY=...` next to "
            "app.py so you don't have to paste it here every time."
        )
        api_key = st.text_input(
            "GOOGLE_API_KEY",
            type="password",
            help="Get one at https://aistudio.google.com/app/apikey",
        )

if not api_key:
    st.info("Enter your Google API key in the sidebar to get started.")
    st.stop()


# ------------------------------------------------------------
# Cached model loading — these are expensive, load once per session
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def load_llm(_api_key: str):
    # Leading underscore on the param tells st.cache_resource not to hash it
    # (API keys shouldn't be used as cache keys); caching is still fine here
    # since the key doesn't change within a session.
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=_api_key,
        temperature=0.2,
        max_output_tokens=512,
    )


@st.cache_resource(show_spinner=False)
def build_vectorstore(pdf_bytes: bytes):
    """Load, split, and embed a PDF. Cached per file content, so re-uploading
    the same PDF (or asking multiple questions) doesn't redo this work."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)

    if not docs:
        return None, 0

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    ).split_documents(docs)

    embeddings = load_embeddings()
    db = FAISS.from_documents(chunks, embeddings)

    return db, len(chunks)


def extract_text(content) -> str:
    """Newer Gemini models (e.g. gemini-3.8-flash) return AIMessage.content
    as a list of structured blocks (e.g. [{"type": "text", "text": "...",
    "extras": {"signature": "..."}}]) rather than a plain string. Pull out
    just the actual text so the UI doesn't render the raw Python objects."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    return str(content)


def answer_question(db, question: str, api_key: str):
    docs = db.similarity_search(question, k=TOP_K)
    context = "\n".join(d.page_content for d in docs)

    prompt = f"""Answer the question using only the context below. If the
context doesn't contain the answer, say you don't know rather than guessing.

Context:
{context}

Question:
{question}

Answer:"""

    llm = load_llm(api_key)

    # Gemini occasionally returns 503 UNAVAILABLE under high demand — this
    # is temporary on Google's end, not an app bug. Retry a few times with
    # a short backoff before giving up.
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = llm.invoke(prompt)
            return extract_text(response.content), docs
        except ServerError:
            if attempt == max_attempts:
                raise
            time.sleep(2 * attempt)  # 2s, then 4s


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer, sources)

# ------------------------------------------------------------
# Upload
# ------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()

    with st.spinner("Reading and indexing the PDF..."):
        db, num_chunks = build_vectorstore(pdf_bytes)

    if db is None:
        st.error("Couldn't extract any text from that PDF (it may be scanned images without OCR).")
    else:
        st.success(f"Indexed {num_chunks} chunks from **{uploaded_file.name}**. Ask away below.")

        st.divider()

        # Show past Q&A for this session
        for q, a, sources in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)
                with st.expander("Sources used"):
                    for i, d in enumerate(sources, 1):
                        page = d.metadata.get("page", "?")
                        st.caption(f"Chunk {i} (page {page})")
                        st.write(d.page_content)

        question = st.chat_input("Ask a question about the PDF")

        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer, sources = answer_question(db, question, api_key)
                    except ServerError:
                        st.error(
                            "Gemini is currently overloaded (503 UNAVAILABLE on "
                            "Google's side). This is temporary — please try asking "
                            "again in a moment."
                        )
                        st.stop()
                st.write(answer)
                with st.expander("Sources used"):
                    for i, d in enumerate(sources, 1):
                        page = d.metadata.get("page", "?")
                        st.caption(f"Chunk {i} (page {page})")
                        st.write(d.page_content)

            st.session_state.chat_history.append((question, answer, sources))
else:
    st.info("Upload a PDF to get started.")
    st.session_state.chat_history = []
