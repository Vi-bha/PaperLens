# 🔍 PaperLens — RAG-Based Research Intelligence

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-blue)](https://huggingface.co/spaces/Vi-bha/PaperLens)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vi-bha/PaperLens/blob/main/PaperLens.ipynb)

> Upload any research paper and instantly get summaries, Q&A, interview prep, implementation roadmaps and critical analysis — powered by RAG + Groq LLaMA 3.1.

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 📄 Summarization | Concise summary of any research paper |
| 🧠 Semantic Q&A | Ask any question, get contextual answers |
| 🎯 Interview Prep | Auto-generated interview questions + answers |
| 🛠️ Implementation Roadmap | Step-by-step plan to implement the paper |
| 🔍 Critical Analysis | Strengths, weaknesses and future directions |

## 🏗️ Architecture
```
PDF Upload → PyMuPDF Parsing → Chunking (500 words, 50 overlap)
    → SentenceTransformer Embeddings → FAISS Vector Store
        → Semantic Retrieval → Groq LLaMA 3.1 → Response
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Vector Store | FAISS |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| LLM | Groq LLaMA 3.1 (8B Instant) |
| RAG Framework | LangChain |
| PDF Parsing | PyMuPDF |
| UI | Gradio |

## ⚙️ How to Run
```bash
# Install dependencies
pip install faiss-cpu sentence-transformers groq langchain gradio pymupdf

# Set your Groq API key (free at console.groq.com)
export GROQ_API_KEY="your_key_here"

# Run
python app.py
```

Or run directly in Google Colab using the badge above.

## 📦 Requirements
```
groq
faiss-cpu
sentence-transformers
pymupdf
langchain
langchain-community
gradio
```

## 🔗 Links
- 🤗 Live Demo: [huggingface.co/spaces/Vi-bha/PaperLens](https://huggingface.co/spaces/Vi-bha/PaperLens)
- 🧬 ResearchMind: [huggingface.co/spaces/Vi-bha/ResearchMind](https://huggingface.co/spaces/Vi-bha/ResearchMind)
- 🔬 MedLens: [huggingface.co/spaces/Vi-bha/MedLens](https://huggingface.co/spaces/Vi-bha/MedLens)

---
*Built by Vibhavari Tummewar | MTech Advanced Computing, MANIT Bhopal*
