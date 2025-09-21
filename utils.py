import re
from typing import List
from PyPDF2 import PdfReader

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = []
    for p in reader.pages:
        text.append(p.extract_text() or '')
    return '\n'.join(text)

def simple_clean(text: str) -> str:
    text = text.replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, max_chars=800) -> List[str]:
    parts = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    for p in parts:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            for i in range(0, len(p), max_chars):
                chunks.append(p[i:i+max_chars])
    if not chunks:
        chunks = [text]
    return chunks