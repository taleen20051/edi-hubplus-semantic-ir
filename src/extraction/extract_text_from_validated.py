from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import requests
from tqdm import tqdm

# PDF extraction
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract_text

# HTML extraction
import trafilatura
from bs4 import BeautifulSoup


# =========================
# CONFIG
# =========================
IN_VALIDATED = Path("data/processed/resources_validated_for_extraction.csv")

CACHE_DIR = Path("data/cache")  # downloaded files stored here for reproducibility
OUT_JSONL = Path("data/processed/resources_text.jsonl")
OUT_MANIFEST = Path("data/processed/extraction_manifest.csv")
TXT_DIR = Path("data/processed/text_by_id")  # human-readable per-resource exports

MIN_TEXT_CHARS = 400  # <-- your choice
TIMEOUT_SECONDS = 25
MAX_RETRIES = 2
SLEEP_BETWEEN_RETRIES_SECONDS = 1.0

ID_COL = "ID"
TITLE_COL = "Title"
URL_COL = "final_url"  # manifest provides final_url; if empty we'll fallback to Link
FALLBACK_URL_COL = "Link"
TYPE_COL = "detected_type"  # pdf or html


# =========================
# Helpers
# =========================
@dataclass
class ExtractionRow:
    ID: str
    Title: str
    url: str
    detected_type: str
    cache_path: str
    download_ok: bool
    http_status_code: Optional[int]
    extraction_ok: bool
    extraction_method: str
    text_length: int
    too_short: bool
    error: str


def safe_filename_from_url(url: str) -> str:
    """Create a stable filename from URL via hash."""
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return h


def normalise_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def download_to_cache(session: requests.Session, url: str, detected_type: str) -> Tuple[bool, Optional[int], Optional[Path], str]:
    """
    Download the resource into CACHE_DIR and return:
    (download_ok, status_code, cache_path, error)
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ext = ".pdf" if detected_type == "pdf" else ".html"
    fname = safe_filename_from_url(url) + ext
    path = CACHE_DIR / fname

    # Cache hit
    if path.exists() and path.stat().st_size > 0:
        return True, None, path, ""

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            r = session.get(url, allow_redirects=True, timeout=TIMEOUT_SECONDS, stream=True)
            status = r.status_code

            # Save regardless of type; we store bytes for PDFs and html bytes for HTML pages
            if status >= 200 and status < 400:
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)
                r.close()

                # Basic sanity check
                if path.exists() and path.stat().st_size > 0:
                    return True, status, path, ""
                else:
                    last_error = "downloaded_file_empty"
            else:
                last_error = f"http_status_{status}"

            try:
                r.close()
            except Exception:
                pass

        except Exception as e:
            last_error = str(e)

        time.sleep(SLEEP_BETWEEN_RETRIES_SECONDS)

    return False, None, None, last_error


# =========================
# Extraction: PDF
# =========================
def extract_pdf_pymupdf(pdf_path: Path) -> str:
    text_parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts)


def extract_pdf_pdfminer(pdf_path: Path) -> str:
    return pdfminer_extract_text(str(pdf_path))


def extract_pdf_text(pdf_path: Path) -> Tuple[bool, str, str]:
    """
    Return (ok, text, method)
    """
    # Try PyMuPDF first
    try:
        t = extract_pdf_pymupdf(pdf_path)
        t = normalise_whitespace(t)
        if len(t) >= 50:  # basic sanity: something came out
            return True, t, "pymupdf"
    except Exception:
        pass

    # Fallback to pdfminer
    try:
        t = extract_pdf_pdfminer(pdf_path)
        t = normalise_whitespace(t)
        if len(t) >= 50:
            return True, t, "pdfminer"
        return False, t, "pdfminer"
    except Exception as e:
        return False, "", f"pdfminer_error:{e}"


# =========================
# Extraction: HTML
# =========================
def extract_html_trafilatura(html_bytes: bytes, url: str) -> str:
    downloaded = html_bytes.decode("utf-8", errors="ignore")
    extracted = trafilatura.extract(downloaded, url=url, include_comments=False, include_tables=True)
    return extracted or ""


def extract_html_bs4(html_bytes: bytes) -> str:
    soup = BeautifulSoup(html_bytes, "lxml")
    # remove obvious boilerplate
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return text


def extract_html_text(html_path: Path, url: str) -> Tuple[bool, str, str]:
    """
    Return (ok, text, method)
    """
    html_bytes = html_path.read_bytes()

    # Try trafilatura first (best main-text extraction)
    try:
        t = extract_html_trafilatura(html_bytes, url)
        t = normalise_whitespace(t)
        if len(t) >= 50:
            return True, t, "trafilatura"
    except Exception:
        pass

    # Fallback to BeautifulSoup
    try:
        t = extract_html_bs4(html_bytes)
        t = normalise_whitespace(t)
        if len(t) >= 50:
            return True, t, "beautifulsoup"
        return False, t, "beautifulsoup"
    except Exception as e:
        return False, "", f"beautifulsoup_error:{e}"


# =========================
# Main
# =========================
def main() -> None:
    if not IN_VALIDATED.exists():
        raise FileNotFoundError(
            f"Validated set not found: {IN_VALIDATED}\n"
            "Run filter_validated_resources.py first."
        )

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    TXT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_VALIDATED, dtype=object)

    # Expected columns
    for col in [ID_COL, TITLE_COL, TYPE_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Columns: {list(df.columns)}")

    # URL column handling (final_url might be empty, so fallback to Link)
    if URL_COL not in df.columns and FALLBACK_URL_COL not in df.columns:
        raise ValueError("No URL column found. Expected 'final_url' or 'Link'.")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "edi-hubplus-dissertation-bot/1.0 (text extraction)",
            "Accept": "*/*",
        }
    )

    extraction_rows = []

    # Open JSONL for writing (overwrite each run)
    with open(OUT_JSONL, "w", encoding="utf-8") as out_f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            rid = str(row.get(ID_COL, "")).strip()
            title = str(row.get(TITLE_COL, "")).strip()
            detected_type = str(row.get(TYPE_COL, "")).strip().lower()

            url = str(row.get(URL_COL, "")).strip()
            if not url:
                url = str(row.get(FALLBACK_URL_COL, "")).strip()

            if not url:
                extraction_rows.append(
                    ExtractionRow(
                        ID=rid,
                        Title=title,
                        url="",
                        detected_type=detected_type,
                        cache_path="",
                        download_ok=False,
                        http_status_code=None,
                        extraction_ok=False,
                        extraction_method="",
                        text_length=0,
                        too_short=True,
                        error="missing_url",
                    )
                )
                continue

            # 1) Download / cache
            download_ok, status_code, cache_path, dl_error = download_to_cache(session, url, detected_type)

            if not download_ok or cache_path is None:
                extraction_rows.append(
                    ExtractionRow(
                        ID=rid,
                        Title=title,
                        url=url,
                        detected_type=detected_type,
                        cache_path="",
                        download_ok=False,
                        http_status_code=status_code,
                        extraction_ok=False,
                        extraction_method="",
                        text_length=0,
                        too_short=True,
                        error=f"download_failed:{dl_error}",
                    )
                )
                continue

            # 2) Extract text
            extraction_ok = False
            text = ""
            method = ""

            try:
                if detected_type == "pdf":
                    extraction_ok, text, method = extract_pdf_text(cache_path)
                elif detected_type == "html":
                    extraction_ok, text, method = extract_html_text(cache_path, url)
                else:
                    extraction_ok = False
                    text = ""
                    method = "unsupported_type"
            except Exception as e:
                extraction_ok = False
                text = ""
                method = "exception"
                dl_error = str(e)

            text = normalise_whitespace(text)
            text_len = len(text)
            too_short = text_len < MIN_TEXT_CHARS

            # Write JSONL even if too_short (useful for debugging), but you can filter later
            record = {
                "id": rid,
                "title": title,
                "url": url,
                "detected_type": detected_type,
                "extraction_method": method,
                "text_length": text_len,
                "too_short": too_short,
                "extracted_text": text,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Also export a human-readable .txt per resource (only if extraction succeeded and text is not too short)
            if extraction_ok and (not too_short):
                txt_path = TXT_DIR / f"{rid}.txt"
                txt_header = (
                    f"ID: {rid}\n"
                    f"Title: {title}\n"
                    f"URL: {url}\n"
                    f"Detected type: {detected_type}\n"
                    f"Extraction method: {method}\n"
                    f"Text length: {text_len}\n"
                    + ("-" * 40)
                    + "\n\n"
                )
                txt_path.write_text(txt_header + text, encoding="utf-8")

            extraction_rows.append(
                ExtractionRow(
                    ID=rid,
                    Title=title,
                    url=url,
                    detected_type=detected_type,
                    cache_path=str(cache_path),
                    download_ok=True,
                    http_status_code=status_code,
                    extraction_ok=extraction_ok,
                    extraction_method=method,
                    text_length=text_len,
                    too_short=too_short,
                    error="" if extraction_ok else (dl_error or "extraction_failed"),
                )
            )

    # Save extraction manifest
    out_df = pd.DataFrame([asdict(r) for r in extraction_rows])
    out_df.to_csv(OUT_MANIFEST, index=False)

    # Print summary for your notes
    print("\n Extraction complete")
    print(f"JSONL saved: {OUT_JSONL}")
    print(f"Manifest saved: {OUT_MANIFEST}")
    print(f"Per-resource TXT saved in: {TXT_DIR}")
    print("\nCounts (extraction_ok):")
    print(out_df["extraction_ok"].value_counts(dropna=False).to_string())
    print("\nCounts (too_short):")
    print(out_df["too_short"].value_counts(dropna=False).to_string())
    print("\nCounts by detected_type:")
    print(out_df["detected_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()