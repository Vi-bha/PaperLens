"""
PaperLens — RAG Pipeline
Handles PDF ingestion, chunking, FAISS indexing, retrieval, and LLM querying.
"""

import os
import numpy as np
import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq


class PaperLensPipeline:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []

    # ─────────────────────────────────────────────
    # Ingestion
    # ─────────────────────────────────────────────
    def load_pdf(self, pdf_path: str) -> str:
        """Extract full text from a PDF file."""
        doc = fitz.open(pdf_path)
        return "".join(page.get_text() for page in doc)

    def chunk_text(self, text: str,
                   chunk_size: int = 500,
                   overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks for retrieval."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        return splitter.split_text(text)

    # ─────────────────────────────────────────────
    # Vector Store
    # ─────────────────────────────────────────────
    def build_index(self, chunks: list[str]) -> None:
        """Embed chunks and build FAISS index."""
        print("Building vector store...")
        self.chunks = chunks
        embeddings = self.embedder.encode(
            chunks, show_progress_bar=False
        ).astype("float32")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"✅ {self.index.ntotal} chunks indexed")

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Return top-k relevant chunks for a query."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        q_emb = self.embedder.encode([query]).astype("float32")
        _, idxs = self.index.search(q_emb, top_k)
        return [self.chunks[i] for i in idxs[0]]

    # ─────────────────────────────────────────────
    # LLM
    # ─────────────────────────────────────────────
    def _llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )
        return response.choices[0].message.content

    def _rag_query(self, question: str) -> str:
        context = "\n\n".join(self.retrieve(question))
        prompt = f"""You are an expert AI researcher.
Answer based ONLY on the provided context from the research paper.

Context:
{context}

Question: {question}

Give a clear, detailed answer."""
        return self._llm(prompt)

    # ─────────────────────────────────────────────
    # 5 Downstream Tasks
    # ─────────────────────────────────────────────
    def summarize(self) -> str:
        return self._rag_query(
            "What is this paper about? Summarize the main contributions, "
            "methodology, and key results."
        )

    def answer(self, question: str) -> str:
        return self._rag_query(question)

    def interview_prep(self) -> str:
        return self._rag_query(
            "Generate 5 technical interview questions with detailed answers "
            "based on this paper's content and methodology."
        )

    def implementation_roadmap(self) -> str:
        return self._rag_query(
            """Create a step-by-step implementation roadmap for a developer
who wants to reproduce this paper from scratch in PyTorch. Include:
1. Prerequisites
2. Step-by-step implementation plan
3. Key components to build
4. Common mistakes to avoid"""
        )

    def critical_analysis(self) -> str:
        return self._rag_query(
            """Critically analyze this paper:
1. KEY STRENGTHS — what does it do well?
2. LIMITATIONS — what are the weaknesses?
3. FUTURE IMPROVEMENTS — what could be done next?
4. REAL WORLD APPLICATIONS — where can this be applied?"""
        )
