# 🔍 PaperLens — AI Research Paper Intelligence

[![HuggingFace Demo](https://img.shields.io/badge/🤗%20Demo-Live-blue)](https://huggingface.co/spaces/Vi-bha/PaperLens)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Upload any research paper PDF → get instant summaries, Q&A, interview prep,
implementation roadmaps, and critical analysis — powered by RAG + LLaMA 3.1.

---

## Pipeline

```
PDF Upload
  → Text Extraction     (PyMuPDF)
  → Chunking            (LangChain RecursiveCharacterTextSplitter, 500 tokens, 50 overlap)
  → Embedding           (SentenceTransformers all-MiniLM-L6-v2)
  → FAISS Indexing      (IndexFlatL2, top-5 semantic retrieval)
  → LLM Generation      (Groq LLaMA 3.1 8B)
```

## 5 Downstream Tasks

| Task | Description |
|------|-------------|
| 📄 Summarize | Main contributions, methodology, key results |
| 💬 Q&A | Ask any question, answered from paper context only |
| 🎯 Interview Prep | 5 technical Q&A pairs based on paper content |
| 🗺️ Implementation Roadmap | Step-by-step PyTorch reproduction guide |
| 🔬 Critical Analysis | Strengths, limitations, future directions |

## Folder Structure

```
PaperLens/
├── app.py            ← Gradio UI entry point
├── pipeline.py       ← RAG pipeline logic
├── requirements.txt
└── README.md
```

## Quickstart

```bash
git clone https://github.com/Vi-bha/PaperLens
cd PaperLens
pip install -r requirements.txt

export GROQ_API_KEY=your_key_here   # free at console.groq.com
python app.py
```

Open `http://localhost:7860` in your browser.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| PDF Parsing | PyMuPDF (fitz) |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | SentenceTransformers all-MiniLM-L6-v2 |
| Vector Store | FAISS IndexFlatL2 |
| LLM | Groq LLaMA 3.1 8B Instant |
| UI | Gradio 4.x |

## Related Projects

| Project | Description | Demo |
|---------|-------------|------|
| [ResearchMind](https://github.com/Vi-bha/ResearchMind) | Autonomous AI scientist querying PubMed 35M+ papers | [🤗 Live](https://huggingface.co/spaces/Vi-bha/ResearchMind) |
| [MedLens](https://github.com/Vi-bha/MedLens) | Multimodal AI for prostate MRI analysis | [🤗 Live](https://huggingface.co/spaces/Vi-bha/MedLens) |

## Author

**Vibhavari Tummewar** — M.Tech Advanced Computing, MANIT Bhopal
Scopus-indexed researcher in LLM-assisted medical AI systems.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/vibhavari-tummewar)
[![HuggingFace](https://img.shields.io/badge/🤗-Vi--bha-yellow)](https://huggingface.co/Vi-bha)
