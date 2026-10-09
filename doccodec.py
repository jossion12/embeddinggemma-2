# -*- coding: utf-8 -*-
"""文档文本提取：把 txt/md/pdf/docx 统一提取为纯文本。

文档入库时的编码格式遵循 EmbeddingGemma 2 规范：
    title: {文件名} | text: {内容}
由调用方组装，这里只负责提取纯文本和生成摘要片段。
"""
from pathlib import Path

DOC_EXTS = {".txt", ".md", ".pdf", ".docx"}
MAX_CHARS = 24000  # 约 6~8k tokens，控制在模型 8K 上下文内
SNIPPET_LEN = 200


def extract_text(path: Path) -> str:
    """提取文档纯文本（超长截断到 MAX_CHARS）。"""
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")[:MAX_CHARS]
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        parts: list[str] = []
        total = 0
        for page in reader.pages:
            t = page.extract_text() or ""
            parts.append(t)
            total += len(t)
            if total > MAX_CHARS:
                break
        return "\n".join(parts)[:MAX_CHARS]
    if ext == ".docx":
        import docx

        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)[:MAX_CHARS]
    raise ValueError(f"不支持的文档格式: {ext}")


def make_snippet(text: str) -> str:
    """生成单行的摘要片段用于搜索结果展示。"""
    return " ".join(text.split())[:SNIPPET_LEN]
