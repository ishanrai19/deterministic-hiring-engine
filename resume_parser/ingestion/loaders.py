"""L0 - ingest. Turns a PDF file or a CSV row into layout tokens + clean text."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import fitz  # PyMuPDF


@dataclass
class Token:
    text: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    line_key: tuple
    start: int = 0
    end: int = 0


@dataclass
class Document:
    candidate_id: str
    source_path: str
    content_hash: str
    text: str = ""
    tokens: List[Token] = field(default_factory=list)
    has_layout: bool = True
    page_count: int = 1
    errors: List[str] = field(default_factory=list)


def _candidate_id(h: str) -> str:
    """Byte-level identity.

    NOTE: this makes reprocessing of *identical bytes* idempotent. It is NOT
    duplicate detection - the same resume re-exported, or with different PDF
    metadata, hashes differently. Content-level dedup needs a separate
    fingerprint over normalised text.
    """
    return "CAND_" + h[:8].upper()


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_pdf(path: str | Path) -> Document:
    path = Path(path)
    data = path.read_bytes()
    h = _hash_bytes(data)
    doc = Document(_candidate_id(h), str(path), h)

    with fitz.open(stream=data, filetype="pdf") as pdf:
        doc.page_count = pdf.page_count
        buf: List[str] = []
        cursor = 0
        for pno in range(pdf.page_count):
            words = pdf[pno].get_text("words")
            words.sort(key=lambda w: (w[5], w[6], w[7]))
            prev: Optional[tuple] = None
            for x0, y0, x1, y1, word, bno, lno, _wno in words:
                if not word.strip():
                    continue
                line_key = (pno + 1, bno, lno)
                if prev is not None:
                    buf.append("\n" if line_key != prev else " ")
                    cursor += 1
                start = cursor
                buf.append(word)
                cursor += len(word)
                doc.tokens.append(Token(word, pno + 1, x0, y0, x1, y1, line_key, start, cursor))
                prev = line_key
            buf.append("\n")
            cursor += 1
        doc.text = "".join(buf)

    if len(doc.text.strip()) < 40:
        doc.errors.append("NO_TEXT_LAYER: digital text extraction returned almost nothing")
    return doc


def _synthesise_tokens(text: str) -> List[Token]:
    tokens: List[Token] = []
    for m in re.finditer(r"\S+", text):
        line_no = text.count("\n", 0, m.start())
        tokens.append(Token(m.group(), 1, 0.0, 0.0, 0.0, 0.0, (1, 0, line_no), m.start(), m.end()))
    return tokens


def load_csv(path, text_column=None, id_column=None, limit=None) -> Iterator[Document]:
    import pandas as pd

    path = Path(path)
    frame = pd.read_csv(path, dtype=str).fillna("")
    if limit:
        frame = frame.head(limit)
    if text_column is None:
        for cand in ("resume_text", "Resume", "resume", "text", "Resume_str", "content"):
            if cand in frame.columns:
                text_column = cand
                break
    if text_column is None:
        text_column = max(frame.columns, key=lambda c: frame[c].str.len().mean())

    for idx, row in frame.iterrows():
        raw = str(row[text_column]).strip()
        if raw.lower().endswith(".pdf") and Path(raw).exists():
            yield load_pdf(raw)
            continue
        h = _hash_bytes(raw.encode("utf-8"))
        cid = str(row[id_column]).strip() if id_column and row.get(id_column) else _candidate_id(h)
        doc = Document(cid, f"{path}#row={idx}", h, raw, _synthesise_tokens(raw), has_layout=False)
        if len(raw) < 40:
            doc.errors.append("EMPTY_ROW: resume text column is effectively empty")
        yield doc
