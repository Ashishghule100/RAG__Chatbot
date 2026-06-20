import os
import uuid
import base64
import tempfile
import shutil
import subprocess
import re
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.vector_store import VectorStore

class DocumentProcessor:
    def __init__(self, use_ocr: bool = True, chunk_size: int = 500, chunk_overlap: int = 100):
        self.use_ocr = use_ocr
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Open‑source embedding model (English)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_store = VectorStore()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    # ---- Helper methods ----

    def extract_grade_from_filename(self, filepath: str) -> str:
        def roman_to_int(s):
            m = {'i':1, 'ii':2, 'iii':3, 'iv':4, 'v':5, 'vi':6, 'vii':7, 'viii':8, 'ix':9, 'x':10}
            return m.get(s.lower())
        p = Path(filepath)
        candidates = [p.name] + [str(part) for part in p.parents[:2]]
        for text in candidates:
            m = re.search(r'(?:class|std|standard)\s*[:\-]?\s*([0-9]{1,2})', text, re.I)
            if m: return m.group(1)
            m = re.search(r'\b([1-9]|10|11|12)\b', text)
            if m: return m.group(1)
            m = re.search(r'\b(i{1,3}|iv|v|vi|vii|viii|ix|x)\b', text, re.I)
            if m:
                val = roman_to_int(m.group(1))
                if val: return str(val)
        return "Unknown"

    def ocr_image(self, image: Image.Image) -> str:
        try:
            image = image.convert("L")
            text = pytesseract.image_to_string(image, lang="eng")
            return text.strip()
        except Exception as e:
            print(f"OCR error: {e}")
            return ""

    # ---- Extraction methods ----

    def extract_text_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        pages = []
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                pages.append({
                    "page_number": i,
                    "text_content": text.strip(),
                    "content_type": "text_only"
                })
        return pages

    def extract_pages_with_ocr(self, pdf_path: str) -> List[Dict[str, Any]]:
        pages_data = []
        try:
            images = convert_from_path(pdf_path, dpi=150)
            for i, img in enumerate(images, start=1):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    img.save(tmp.name, "PNG", optimize=True)
                    with open(tmp.name, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode('utf-8')
                    os.remove(tmp.name)
                text = self.ocr_image(img)
                pages_data.append({
                    "page_number": i,
                    "text_content": text,
                    "image_base64": b64,
                    "content_type": "ocr_page"
                })
                img.close()
        except Exception as e:
            print(f"OCR extraction error: {e}")
        return pages_data

    def extract_text_txt(self, txt_path: str) -> List[Dict[str, Any]]:
        with open(txt_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        start = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end = "*** END OF THE PROJECT GUTENBERG EBOOK"
        if start in full_text and end in full_text:
            full_text = full_text.split(start)[1].split(end)[0]
        full_text = re.sub(r'\n\s*\n', '\n\n', full_text).strip()
        return [{
            "page_number": 1,
            "text_content": full_text,
            "content_type": "txt_page"
        }]

    def extract_text_docx(self, docx_path: str) -> List[Dict[str, Any]]:
        try:
            import docx
            doc = docx.Document(docx_path)
            full = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full.append(para.text.strip())
            text = '\n\n'.join(full)
            return [{
                "page_number": 1,
                "text_content": text.strip(),
                "content_type": "docx_text"
            }]
        except Exception as e:
            print(f"DOCX error: {e}")
            return []

    def convert_docx_to_pdf(self, docx_path: str):
        # kept for compatibility but we don't use it for TXT
        return None

    def extract_raw_pages(self, file_path: str, use_ocr: bool) -> List[Dict[str, Any]]:
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            return self.extract_pages_with_ocr(file_path) if use_ocr else self.extract_text_pdf(file_path)
        elif ext == '.docx':
            return self.extract_text_docx(file_path)
        elif ext == '.txt':
            return self.extract_text_txt(file_path)
        else:
            print(f"⚠️ Unsupported: {ext}")
            return []

    # ---- Chunking and embedding ----

    def chunk_page(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = page_data.get("text_content", "")
        if not text.strip():
            return [page_data.copy()]
        splits = self.text_splitter.split_text(text)
        chunks = []
        for idx, chunk_text in enumerate(splits):
            chunk = page_data.copy()
            chunk["chunk_index"] = idx
            chunk["text_content"] = chunk_text
            chunks.append(chunk)
        return chunks

    def embed_text(self, text: str) -> List[float]:
        # SentenceTransformer returns numpy array
        return self.embed_model.encode(text).tolist()

    # ---- Main processing ----

    def process_single_document(self, file_path: str, force_reprocess: bool = False) -> bool:
        bookname = os.path.basename(file_path)
        if not force_reprocess and self.book_exists(bookname):
            print(f"⚠️ {bookname} already exists. Use --force.")
            return False

        pages = self.extract_raw_pages(file_path, self.use_ocr)
        if not pages:
            print(f"❌ No content from {bookname}")
            return False

        grade = self.extract_grade_from_filename(file_path)

        all_docs = []
        for page in pages:
            chunks = self.chunk_page(page)
            for chunk in chunks:
                text = chunk["text_content"]
                if not text.strip():
                    continue
                embedding = self.embed_text(text)
                chunk_id = str(uuid.uuid4())
                doc = {
                    "chunk_id": chunk_id,
                    "text_content": text,
                    "embedding": embedding,
                    "bookname": bookname,
                    "grade": grade,
                    "page_no": chunk["page_number"],
                    "content_type": chunk.get("content_type", "text"),
                    "image_base64": chunk.get("image_base64", ""),
                    "image_description": chunk.get("image_description", ""),
                    "subject": "Unknown",
                    "board": "Unknown"
                }
                all_docs.append(doc)

        success = 0
        for doc in all_docs:
            if self.vector_store.upsert_chunk(doc):
                success += 1
        print(f"✅ Stored {success} chunks from {bookname}")
        return success > 0

    def process_directory(self, directory: str):
        supported = ('.pdf', '.docx', '.txt')
        files = []
        for root, _, fs in os.walk(directory):
            for f in fs:
                if f.lower().endswith(supported):
                    files.append(os.path.join(root, f))
        print(f"📂 Found {len(files)} files")
        for f in files:
            self.process_single_document(f, force_reprocess=True)

    def process_single_pdf(self, file_path: str, force_reprocess: bool = False) -> bool:
        return self.process_single_document(file_path, force_reprocess)

    def book_exists(self, bookname: str) -> bool:
        results = self.vector_store.search([0.0]*384, top_k=1, filters={"bookname": bookname})
        return len(results) > 0

    def close(self):
        self.vector_store.close()