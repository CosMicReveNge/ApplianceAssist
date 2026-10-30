# 🏠 ApplianceAssist AI
### A RAG-Powered Home Appliance Manual and Troubleshooting Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io)
[![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-orange)](https://ai.google.dev)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green)](https://github.com/facebookresearch/faiss)

---

## 📖 Project Description

**ApplianceAssist AI** is a document-based AI assistant that allows users to
upload home appliance manuals in PDF format and ask questions about them using
natural language.

The system retrieves the most relevant sections from the uploaded manuals and
passes them to Google Gemini to generate precise, grounded answers — without
inventing information that is not in the document.

---

## 🎯 Problem Statement

Appliance manuals are often long, technical, and difficult to navigate.
Users spend significant time searching for error codes, maintenance schedules,
or troubleshooting procedures. ApplianceAssist AI solves this by letting users
ask plain-English questions and get instant, cited answers drawn directly from
the uploaded manual.

---

## 🏆 Objectives

1. Build a functional RAG pipeline using open-source and free-tier tools.
2. Allow users to upload one or more PDF manuals without coding knowledge.
3. Retrieve the most relevant manual sections using semantic similarity.
4. Generate grounded answers using Google Gemini with strict context adherence.
5. Display source citations (filename + page number) for every answer.
6. Deploy the application publicly on Render.

---

## ✨ Features

| Feature | Details |
|---|---|
| Multi-PDF upload | Upload one or more appliance manuals at once |
| Intelligent chunking | RecursiveCharacterTextSplitter with metadata |
| Semantic search | FAISS vector store with cosine similarity |
| Grounded LLM answers | Gemini answers only from retrieved context |
| Source citations | Filename + page number for every answer |
| Relevance threshold | Refuses to answer when context is irrelevant |
| Error handling | Clear messages for all failure scenarios |
| Session memory | Chat history preserved during the session |
| Deployment-ready | Configured for Render free tier |

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Frontend | Streamlit 1.35 |
| PDF extraction | PyPDF 4.2 |
| Text splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector database | FAISS (in-memory, CPU) |
| LLM | Google Gemini 1.5 Flash |
| Environment | python-dotenv |
| Deployment | Render |

---

## 🏗️ RAG Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      INDEXING PHASE                             │
│                                                                 │
│  PDF Upload → PyPDF Extraction → Text Chunking                  │
│                                  (1000 chars, 200 overlap)      │
│                                        │                        │
│                                        ▼                        │
│                              Sentence Transformer               │
│                              (all-MiniLM-L6-v2)                 │
│                                        │                        │
│                                        ▼                        │
│                              FAISS In-Memory Index              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      QUERY PHASE                                │
│                                                                 │
│  User Question → Sentence Transformer → Query Embedding         │
│                                              │                  │
│                                              ▼                  │
│                              FAISS Similarity Search            │
│                              (Top-5, threshold ≥ 30%)           │
│                                              │                  │
│                                    ┌─────────┴──────────┐      │
│                                    │  Relevant chunks   │      │
│                                    │  found?            │      │
│                                    └─────────┬──────────┘      │
│                           YES ◄──────────────┤                  │
│                            │                 │ NO               │
│                            ▼                 ▼                  │
│                    Build Prompt         "No information"         │
│                    + Context            message                  │
│                            │                                    │
│                            ▼                                    │
│                    Google Gemini 1.5 Flash                      │
│                    (system-prompted to stay on context)          │
│                            │                                    │
│                            ▼                                    │
│                    Answer + Source Citations                     │
└─────────────────────────────────────────────────────────────────┘
```

### RAG Pipeline Explained (for Viva)

**RAG** stands for **Retrieval-Augmented Generation**. Instead of asking an
LLM to answer from its training data (which may hallucinate), we:

1. **Index the documents**: Convert the manual text into numerical vectors
   (embeddings) that capture the *meaning* of each passage.

2. **Retrieve relevant passages**: When the user asks a question, we convert
   the question into an embedding and find the passages most *semantically
   similar* to it using FAISS (a fast vector similarity library).

3. **Augment the prompt**: We paste the retrieved passages into the prompt as
   context, telling Gemini: "Answer the question using ONLY this text."

4. **Generate a grounded answer**: Gemini reads the context and the question
   and produces an answer. If the context doesn't contain the answer, it says
   so rather than guessing.

**Why is this better than just asking ChatGPT?**
- The LLM cannot hallucinate manual-specific error codes or specs.
- Answers are traceable to specific pages of the uploaded document.
- Works for any appliance, any brand — not limited to training data.

---

## 📁 Project Folder Structure

```
ApplianceAssist/
│
├── app.py              ← Streamlit UI and application workflow
├── rag_pipeline.py     ← Embeddings, FAISS, Gemini, RAG query
├── pdf_processor.py    ← PDF validation, extraction, chunking
├── requirements.txt    ← Pinned Python dependencies
├── .env.example        ← Environment variable template
├── .gitignore          ← Files excluded from version control
├── README.md           ← This file
└── sample_manuals/
    └── README.md       ← Instructions for finding test PDFs
```

---

## 🚀 Installation and Setup (macOS)

### Prerequisites

- Python 3.10 or higher
- A Google Gemini API key (free tier works)

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ApplianceAssist.git
cd ApplianceAssist
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages including the Sentence Transformers
model (which will be downloaded automatically on first use).

### Step 4 — Configure your Gemini API key

```bash
cp .env.example .env
```

Open `.env` in any text editor and replace `your_api_key_here` with your
actual Gemini API key:

```
GEMINI_API_KEY=AIzaSy...your_actual_key_here...
```

**Never commit the `.env` file to GitHub.**

### Step 5 — Run the application

```bash
streamlit run app.py
```

The application will open automatically at `http://localhost:8501`.

---

## 🔑 Obtaining a Gemini API Key

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account.
3. Click **"Create API key"**.
4. Select an existing Google Cloud project or create a new one.
5. Copy the generated API key.
6. Paste it into your `.env` file as shown above.

The **free tier** of Gemini 1.5 Flash is sufficient for this project.

---

## 🧪 Local Testing Guide

### Feature-by-feature testing checklist

| # | Test | Expected Result |
|---|---|---|
| 1 | Upload a valid appliance PDF | Success message, chunk count shown |
| 2 | Check extracted text | No errors, some warnings for blank pages OK |
| 3 | Ask: "What does error code E3 mean?" | Answer referencing the manual |
| 4 | Check source citation | Filename + page number displayed |
| 5 | Ask a question not in the manual | "Insufficient information" message |
| 6 | Upload a non-PDF file | Error: "not a valid PDF file" |
| 7 | Submit an empty question | Warning: "Please enter a question" |
| 8 | Remove API key and ask a question | Clear API key error message |
| 9 | Upload multiple PDFs | All indexed, chunk count increases |
| 10 | Click "Clear All & Reset" | Chat and index cleared, fresh start |

### Sample demonstration questions

Use these during your lab examination:

1. **"Why is my washing machine displaying error E3?"**
2. **"How often should I clean my refrigerator filter?"**
3. **"What does eco mode do and how do I activate it?"**
4. **"How do I troubleshoot a microwave that is not heating?"**
5. **"What safety precautions should I take before cleaning the appliance?"**
6. **"What is the recommended maintenance schedule for this appliance?"**
7. **"Can I repair the compressor myself?"** ← Should recommend a technician

---

## ☁️ Render Deployment

### Prerequisites

- A GitHub account with your project pushed to a public (or private) repository.
- A Render account at [https://render.com](https://render.com) (free tier).

---

### Step 1 — Push to GitHub

```bash
# Inside the ApplianceAssist directory
git init
git add .
git commit -m "Initial commit: ApplianceAssist AI"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ApplianceAssist.git
git push -u origin main
```

---

### Step 2 — Create a Render Web Service

1. Go to [https://dashboard.render.com](https://dashboard.render.com).
2. Click **"New +"** → **"Web Service"**.
3. Click **"Connect account"** to link your GitHub account.
4. Find and select your **ApplianceAssist** repository.
5. Click **"Connect"**.

---

### Step 3 — Configure the Web Service

Fill in the following settings:

| Setting | Value |
|---|---|
| **Name** | `applianceassist-ai` (or any name) |
| **Region** | Oregon (US West) or your nearest region |
| **Branch** | `main` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| **Instance Type** | Free |

---

### Step 4 — Set the Gemini API Key

1. In the Render service settings, go to **"Environment"**.
2. Click **"Add Environment Variable"**.
3. Set:
   - **Key**: `GEMINI_API_KEY`
   - **Value**: `AIzaSy...your_actual_key...`
4. Click **"Save Changes"**.

---

### Step 5 — Deploy

1. Click **"Create Web Service"**.
2. Render will clone your repository, install dependencies, and start the app.
3. The first deployment takes 3–5 minutes (model download included).
4. Once complete, you will see a URL like:
   `https://applianceassist-ai.onrender.com`

---

### Step 6 — Verify the Deployment

1. Open the Render URL in a browser.
2. Upload a PDF appliance manual.
3. Ask one of the sample questions above.
4. Verify that the answer and source citations appear correctly.

> **Note:** On Render's free tier, the service spins down after 15 minutes of
> inactivity. The first request after spin-down takes ~30 seconds.

---

## ⚠️ Known Limitations

1. **Scanned PDFs**: The app cannot extract text from PDFs that are scanned
   images. Only PDFs with selectable digital text are supported.
2. **Free tier memory**: Render's free tier has 512 MB RAM. Very large PDFs
   (hundreds of pages) may cause memory issues.
3. **Session state**: The FAISS index is in-memory and is lost when the
   Render server restarts or the session ends.
4. **API rate limits**: The Gemini free tier has request limits. Heavy usage
   may trigger temporary throttling.
5. **Language**: The embedding model works best with English text.

---

## 🔮 Future Enhancements

- Persistent FAISS index storage using Render Disks or cloud storage.
- OCR support for scanned PDFs using Tesseract.
- Support for additional document formats (DOCX, HTML).
- User authentication for multi-user deployments.
- Answer feedback mechanism (thumbs up/down).
- Multi-language support.
- Hybrid search combining keyword and semantic retrieval.

---

## 📄 License

This project was developed as a college laboratory examination project.
Feel free to use and modify it for educational purposes.
