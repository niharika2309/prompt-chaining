"""PDF -> text extraction (pypdf) with caching.

The 50 documents in pdfs/ are digital (reportlab) PDFs; text extraction is
clean. Scanned variants (pdfs_scanned/) are image-only and are out of scope
for v1 (separate phase using the endpoint's vision input).
"""
import config
from pathlib import Path

from pypdf import PdfReader


def extract_one(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n\n".join(pages)


def extract_all(filenames: list[str] | None = None) -> dict[str, str]:
    """Extract text for the given PDFs (or all) into the corpus cache.

    Returns {pdf_filename: text}. Cached texts in CORPUS_DIR are reused.
    """
    config.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    if filenames is None:
        filenames = sorted(p.name for p in config.PDF_DIR.glob("*.pdf"))

    texts: dict[str, str] = {}
    for name in filenames:
        cache = config.CORPUS_DIR / (Path(name).stem + ".txt")
        if cache.exists() and cache.stat().st_size > 0:
            texts[name] = cache.read_text()
            continue
        text = extract_one(config.PDF_DIR / name)
        cache.write_text(text)
        texts[name] = text
        flag = "OK " if len(text) > 200 else "THIN"
        print(f"  extracted [{flag}] {name} ({len(text)} chars)")
    return texts
