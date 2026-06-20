A fully offline, open‑source RAG (Retrieval-Augmented Generation) system that answers questions from your private collection of PDFs and text books.
No API keys. No cloud dependencies. Everything runs locally.

Key features:

📄 Ingest PDFs, TXT, and DOCX (with OCR for scanned pages)

🧠 Local embeddings + LLM (using llama-cpp-python)

⚡ Fast ANN retrieval with ChromaDB

🖥️ Interactive Web UI (Streamlit)

📊 Evaluation suite (latency + accuracy)


🧠 Tech Stack
Component	Technology
Language	Python 3.11
Embeddings	sentence-transformers/all-MiniLM-L6-v2 (or multilingual)
Vector DB	ChromaDB (persistent, HNSW)
LLM	llama-cpp-python with local .gguf (TinyLlama, Mistral, etc.)
OCR	Tesseract
Chunking	langchain-text-splitters (RecursiveCharacterTextSplitter)
Frontend	Streamlit
Evaluation	Custom script with latency & accuracy metrics


🙏 Acknowledgements
Sentence‑Transformers

ChromaDB

llama-cpp-python

Streamlit

Project Gutenberg for the sample books

Built with ❤️ using open‑source tools.
