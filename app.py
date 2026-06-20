import streamlit as st
import os
from src.rag.rag import RAGSystem
from src.rag.pdf_processor import DocumentProcessor

import streamlit as st
import traceback

try:
    # your existing imports and code
    from src.rag.rag import RAGSystem
    from src.rag.pdf_processor import DocumentProcessor
    # ... rest of app
except Exception as e:
    st.error(f"Error loading app: {e}")
    st.code(traceback.format_exc())

st.set_page_config(page_title="Open‑Source RAG Chatbot", layout="wide")
st.title("📚 RAG Chatbot (Local Open‑Source Models)")

if "rag" not in st.session_state:
    model_path = st.sidebar.text_input("GGUF Model Path", "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    st.session_state.rag = RAGSystem(model_path=model_path)
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("📁 Ingestion")
    uploaded_files = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded_files and st.button("Ingest"):
        processor = DocumentProcessor(use_ocr=True)
        for uploaded in uploaded_files:
            temp_path = f"/tmp/{uploaded.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            processor.process_single_pdf(temp_path, force_reprocess=True)
            os.remove(temp_path)
        processor.close()
        st.success("Ingestion complete!")

    st.header("🔎 Filters")
    filters = {}
    b = st.text_input("Book name")
    if b: filters["bookname"] = b
    g = st.text_input("Grade")
    if g: filters["grade"] = g
    s = st.text_input("Subject")
    if s: filters["subject"] = s

st.subheader("💬 Ask a Question")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.rag.answer(prompt, filters=filters if filters else None)
            answer = result["answer"]
            sources = result.get("sources", [])
            st.markdown(answer)
            if sources:
                with st.expander("📖 Sources"):
                    for s in sources:
                        st.write(f"- **{s['bookname']}**, page {s['page_no']}")
    st.session_state.messages.append({"role": "assistant", "content": answer})