# pre_process.py
import re
import unicodedata
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)


def normalize_text(text: str) -> str:
    """
    RAG-safe normalization:
    - Preserve Unicode meaning
    - Remove PDF/Word artifacts
    - Preserve paragraph structure
    """
    text = unicodedata.normalize("NFKC", text)

    # Remove common footer/header noise
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Clean whitespace but keep paragraphs
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def process_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    for d in docs:
        d.page_content = normalize_text(d.page_content)

    return docs 


def process_word(file_path: str) -> List[Document]:
    loader = UnstructuredWordDocumentLoader(file_path)
    docs = loader.load()

    for d in docs:
        d.page_content = normalize_text(d.page_content)

    return docs 


def process_document(file_path: str) -> List[Document]:
    """
    Load and normalize document.
    Chunking is handled downstream in indexer.py
    """
    if file_path.lower().endswith(".pdf"):
        return process_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        return process_word(file_path)
    else:
        raise ValueError("Only PDF and DOCX are supported")
