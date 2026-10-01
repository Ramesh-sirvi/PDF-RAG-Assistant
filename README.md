# 📄 Chat with Your PDF — RAG App

A **Retrieval-Augmented Generation (RAG)** application built with **Streamlit** that allows users to upload a PDF and ask questions about its content.

The app retrieves relevant sections from the uploaded document and uses **Gemini** to generate answers based on the retrieved context.

## 🔄 How It Works

```text
PDF Upload
    ↓
PyPDFLoader
    ↓
Text Chunking
    ↓
HuggingFace Embeddings
    ↓
FAISS Vector Search
    ↓
Relevant Chunks
    ↓
Gemini
    ↓
Answer
```

## ✨ Features

* Upload and chat with any PDF
* RAG-based question answering
* Local embeddings using `all-MiniLM-L6-v2`
* FAISS similarity search
* Gemini-powered answer generation
* Displays sources used for each answer
* Streamlit chat interface
* Supports configurable Google API key

## 🛠️ Technologies

* Python
* Streamlit
* LangChain
* PyPDF
* Hugging Face Embeddings
* FAISS
* Google Gemini
* Sentence Transformers

## 📁 Project Structure

```text
Chat-with-PDF/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🔑 API Key

Create a Google Gemini API key from Google AI Studio.

Set it as an environment variable:

```bash
GOOGLE_API_KEY=your_api_key
```

**Do not upload your `.env` file or API key to GitHub.**

## ▶️ Run the App

```bash
streamlit run app.py
```

Upload a PDF and start asking questions.

## ⚠️ Limitations

* Currently supports one PDF at a time.
* Scanned/image-only PDFs require OCR.
* FAISS index is stored in memory and is rebuilt when the app restarts.
* Gemini is required for answer generation.

## 🚀 Future Improvements

* Support multiple PDFs
* Add OCR for scanned documents
* Persistent vector database
* Chat history storage
* Cloud deployment

## 👨‍💻 Author

**Ramesh Choudhary**

GitHub: `https://github.com/Ramesh-sirvi`

⭐ If you find this project useful, consider giving it a star.
