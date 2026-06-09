"""
PaperLens — Gradio UI
Run: python app.py
"""

import os
import gradio as gr
from pipeline import PaperLensPipeline

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not set. Export it: export GROQ_API_KEY=your_key"
    )

pipeline = PaperLensPipeline(groq_api_key=GROQ_API_KEY)


def upload_and_process(pdf_file):
    if pdf_file is None:
        return "❌ Please upload a PDF file."
    try:
        text = pipeline.load_pdf(pdf_file.name)
        chunks = pipeline.chunk_text(text)
        pipeline.build_index(chunks)
        return pipeline.summarize()
    except Exception as e:
        return f"❌ Error processing PDF: {e}"


def answer_question(question: str):
    if not question.strip():
        return "❌ Please enter a question."
    if pipeline.index is None:
        return "❌ Please upload a paper first."
    try:
        return pipeline.answer(question)
    except Exception as e:
        return f"❌ Error: {e}"


def run_task(task_fn):
    if pipeline.index is None:
        return "❌ Please upload a paper first."
    try:
        return task_fn()
    except Exception as e:
        return f"❌ Error: {e}"


with gr.Blocks(title="PaperLens — Research Paper Intelligence",
               theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🔍 PaperLens — AI Research Paper Intelligence
    Upload any research paper PDF → get instant summaries, Q&A, interview prep,
    implementation roadmaps, and critical analysis powered by RAG + LLaMA 3.1.
    """)

    with gr.Tab("📄 Upload & Summarize"):
        pdf_input = gr.File(label="Upload Research Paper (PDF)",
                            file_types=[".pdf"])
        upload_btn = gr.Button("Analyze Paper", variant="primary")
        summary_output = gr.Markdown(label="Paper Summary")
        upload_btn.click(upload_and_process, pdf_input, summary_output)

    with gr.Tab("💬 Ask Questions"):
        question_input = gr.Textbox(
            label="Ask anything about the paper",
            placeholder="e.g. What evaluation metrics did they use?"
        )
        ask_btn = gr.Button("Ask")
        answer_output = gr.Markdown(label="Answer")
        ask_btn.click(answer_question, question_input, answer_output)

    with gr.Tab("🎯 Interview Prep"):
        interview_btn = gr.Button("Generate Interview Questions", variant="primary")
        interview_output = gr.Markdown(label="Interview Q&A")
        interview_btn.click(
            lambda: run_task(pipeline.interview_prep), None, interview_output
        )

    with gr.Tab("🗺️ Implementation Roadmap"):
        roadmap_btn = gr.Button("Generate Implementation Roadmap", variant="primary")
        roadmap_output = gr.Markdown(label="How to implement this paper")
        roadmap_btn.click(
            lambda: run_task(pipeline.implementation_roadmap), None, roadmap_output
        )

    with gr.Tab("🔬 Critical Analysis"):
        critique_btn = gr.Button("Analyze Strengths & Weaknesses", variant="primary")
        critique_output = gr.Markdown(label="Critical Analysis")
        critique_btn.click(
            lambda: run_task(pipeline.critical_analysis), None, critique_output
        )

if __name__ == "__main__":
    demo.launch()
