#!/usr/bin/env python3
import os
import argparse
from dotenv import load_dotenv

from src.rag.pdf_processor import DocumentProcessor
from src.rag.rag import RAGSystem

load_dotenv()

def process_command(args):
    use_ocr = not args.no_ocr
    print("🚀 Starting document processing...")
    processor = DocumentProcessor(use_ocr=use_ocr)
    try:
        if args.single:
            processor.process_single_pdf(args.directory, force_reprocess=args.force)
        else:
            processor.process_directory(args.directory)
    finally:
        processor.close()

def query_command(args):
    filters = {}
    if args.bookname: filters["bookname"] = args.bookname
    if args.grade: filters["grade"] = args.grade
    if args.subject: filters["subject"] = args.subject
    rag = RAGSystem(model_path=args.model_path)
    try:
        while True:
            q = input("\n🎯 Question: ").strip()
            if q.lower() in ("exit", "quit"):
                break
            if not q:
                continue
            result = rag.answer(q, filters=filters if filters else None)
            print(f"\n💡 Answer: {result['answer']}")
            if result.get('sources'):
                print("\n📚 Sources:")
                for s in result['sources']:
                    print(f"   - {s['bookname']}, page {s['page_no']}")
    finally:
        rag.close()

def main():
    parser = argparse.ArgumentParser(description="Open‑Source RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Process command
    p = subparsers.add_parser("process", help="Ingest documents")
    p.add_argument("directory", help="Path to file or directory")
    p.add_argument("--single", action="store_true", help="Process single file")
    p.add_argument("--force", action="store_true", help="Force reprocess")
    p.add_argument("--no-ocr", action="store_true", help="Disable OCR")

    # Query command
    q = subparsers.add_parser("query", help="Interactive Q&A")
    q.add_argument("--bookname", help="Filter by book name")
    q.add_argument("--grade", help="Filter by grade")
    q.add_argument("--subject", help="Filter by subject")
    q.add_argument("--model-path", default="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                   help="Path to GGUF model file")

    # UI command
    ui = subparsers.add_parser("ui", help="Launch Streamlit UI")

    args = parser.parse_args()
    if args.command == "process":
        process_command(args)
    elif args.command == "query":
        query_command(args)
    elif args.command == "ui":
        os.system("streamlit run app.py")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()