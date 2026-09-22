"""文档解析与分块。

设计要点（面试可讲）：
1. 分块位置比 chunk_size 更重要：先按 Markdown 标题/段落边界粗切（标题感知），
   再对超长块用 RecursiveCharacterTextSplitter 按中文标点二次切分，
   避免"标题和它的参数被切到两个 chunk"。
2. 每个 chunk 携带 source（文件名）+ 标题上下文（heading_path），
   用于来源引用和检索后展示。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SEPARATORS, CHUNK_SIZE


@dataclass
class Chunk:
    """一个检索单元。metadata 会写入向量库与 BM25，供过滤和来源引用。"""
    text: str
    source: str
    heading_path: str = ""
    chunk_id: str = ""
    metadata: dict = field(default_factory=dict)


# ---------------- 文档解析 ----------------

def parse_document(path: Path) -> str:
    """按扩展名解析文档为纯文本。支持 PDF / DOCX / TXT / MD。"""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(path)
    if ext == ".docx":
        return _parse_docx(path)
    if ext in (".txt", ".md", ".markdown"):
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"不支持的文档类型: {ext}（支持 PDF/DOCX/TXT/MD）")


def _parse_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # 保留页码信息，供来源引用
        pages.append(f"\n[第{i + 1}页]\n{text}")
    return "\n".join(pages)


def _parse_docx(path: Path) -> str:
    import docx
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ---------------- 分块 ----------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _split_by_heading(text: str):
    """按 Markdown 标题粗切，返回 (heading_path, section_text) 列表。

    无标题的文档整体作为一节，heading_path 为空。
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush():
        nonlocal current_lines
        if current_lines:
            sections.append((current_heading, current_lines))
            current_lines = []

    for line in lines:
        m = _HEADING_RE.match(line.strip())
        if m:
            flush()
            # 保留标题层级信息（## 一级标题链）
            current_heading = line.strip().lstrip("#").strip()
        else:
            current_lines.append(line)
    flush()

    return [(h, "\n".join(ls).strip()) for h, ls in sections if "\n".join(ls).strip()]


def chunk_document(path: Path) -> list[Chunk]:
    """完整分块管线：解析 -> 标题感知粗切 -> 递归精切 -> 生成 Chunk。"""
    text = parse_document(path)
    source = path.name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        keep_separator=True,
    )

    chunks: list[Chunk] = []
    for heading, section in _split_by_heading(text):
        if len(section) <= CHUNK_SIZE:
            pieces = [section]
        else:
            pieces = splitter.split_text(section)
        for p in pieces:
            p = p.strip()
            if not p:
                continue
            # 标题并入正文，保证检索时 chunk 自带上下文
            content = f"{heading}\n{p}" if heading else p
            chunks.append(
                Chunk(
                    text=content,
                    source=source,
                    heading_path=heading,
                    metadata={"source": source, "heading": heading},
                )
            )
    return chunks
