import subprocess
import tempfile
from pathlib import Path

import docx2txt
from pypdf import PdfReader

from .bailian_client import BailianClient


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def extract_text_from_docx(path: Path) -> str:
    return normalize_text(docx2txt.process(str(path)) or "")


def extract_text_from_pdf_native(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return normalize_text("\n".join(pages))


def extract_text_from_pdf_vl_ocr(
    path: Path,
    client: BailianClient,
    max_pages: int | None = None,
) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for VL OCR on PDF. Install with: pip install PyMuPDF"
        ) from exc

    doc = fitz.open(str(path))
    page_total = len(doc)
    page_limit = min(page_total, max_pages) if max_pages else page_total

    page_texts: list[str] = []
    for i in range(page_limit):
        page = doc.load_page(i)
        # 2x scale improves OCR accuracy for small fonts.
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        image_bytes = pix.tobytes("png")
        text = client.ocr_image_bytes(
            image_bytes=image_bytes,
            image_mime="image/png",
            prompt=(
                f"这是PDF的第{i + 1}页。请逐行OCR提取全文。"
                "保留条款编号、章节标题、数字与标点，不要解释。"
            ),
        )
        page_texts.append(f"[第{i + 1}页]\n{text}")

    doc.close()
    return normalize_text("\n\n".join(page_texts))


def convert_doc_to_docx_with_word(doc_path: Path) -> Path:
    """Convert .doc to .docx with local Microsoft Word COM automation."""
    if doc_path.suffix.lower() != ".doc":
        raise ValueError("convert_doc_to_docx_with_word only accepts .doc")

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        out_docx = Path(tmp.name)

    doc_ps = str(doc_path).replace("'", "''")
    out_ps = str(out_docx).replace("'", "''")

    script = (
        "$ErrorActionPreference='Stop';"
        "$word=New-Object -ComObject Word.Application;"
        "$word.Visible=$false;"
        f"$doc=$word.Documents.Open('{doc_ps}');"
        f"$doc.SaveAs([ref]'{out_ps}', [ref]16);"
        "$doc.Close();"
        "$word.Quit();"
    )

    subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return out_docx


def extract_text_from_image(path: Path, client: BailianClient) -> str:
    text = client.ocr_image_path(
        str(path),
        prompt="请OCR提取图片中的全部文字，保留原始结构，不要解释。",
    )
    return normalize_text(text)


def extract_text_from_file(
    file_path: str,
    client: BailianClient,
    pdf_mode: str = "vl",
    max_pages: int | None = None,
) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".docx":
        return extract_text_from_docx(path)
    if suffix == ".pdf":
        if pdf_mode == "vl":
            try:
                return extract_text_from_pdf_vl_ocr(path, client, max_pages=max_pages)
            except Exception:
                # Fallback to native parser if VL OCR fails.
                return extract_text_from_pdf_native(path)
        return extract_text_from_pdf_native(path)
    if suffix == ".doc":
        # Best effort: convert with Word then parse as docx.
        converted = convert_doc_to_docx_with_word(path)
        try:
            return extract_text_from_docx(converted)
        finally:
            converted.unlink(missing_ok=True)
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return extract_text_from_image(path, client)

    raise ValueError(f"Unsupported file type: {suffix}")
