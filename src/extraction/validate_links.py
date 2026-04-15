from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import requests


# =========================
# CONFIG
# =========================
IN_CSV = Path("data/processed/resources_included_only.csv")
OUT_MANIFEST = Path("data/processed/resource_manifest.csv")
OUT_LOG = Path("data/processed/link_validation_log.csv")

ID_COL = "ID"
TITLE_COL = "Title"
URL_COL = "Link"

# Network behaviour (methodology-relevant)
TIMEOUT_SECONDS = 15
MAX_RETRIES = 2
SLEEP_BETWEEN_RETRIES_SECONDS = 1.0

# Treat these as “accessible enough to try extraction”
# (Many sites block HEAD but allow GET, so we fallback anyway)
ALLOWED_STATUS_FOR_TRY = {200, 201, 202, 203, 204, 206, 301, 302, 303, 307, 308}

# If content-type is missing/incorrect, we also use URL heuristics
PDF_EXTENSIONS = (".pdf", ".PDF")


# =========================
# DATA STRUCTURES
# =========================
@dataclass
class ValidationLog:
    resource_id: str
    url: str
    attempt: int
    method: str
    ok: bool
    status_code: Optional[int]
    error: str


def normalise_url(url: str) -> str:
    """Basic URL cleanup without being too aggressive."""
    u = str(url).strip()
    # Common Excel issues: trailing spaces, line breaks
    u = u.replace("\n", "").replace("\r", "")
    return u


def detect_type(url: str, content_type: Optional[str]) -> str:
    """Return: 'pdf', 'html', or 'unknown'."""
    ct = (content_type or "").lower().strip()

    # Strong signals from content-type
    if "application/pdf" in ct:
        return "pdf"
    if "text/html" in ct:
        return "html"

    # Weaker signals / fallbacks
    if url.endswith(PDF_EXTENSIONS):
        return "pdf"
    if ct.startswith("text/"):
        return "html"

    return "unknown"


def request_with_fallback(
    session: requests.Session,
    url: str,
    logs: List[ValidationLog],
    resource_id: str,
) -> Tuple[bool, Optional[int], Optional[str], Optional[str], Optional[int]]:
    """
    Try HEAD first, then GET if needed.
    Returns: (ok, status_code, final_url, content_type, content_length)
    """
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 2):  # e.g. MAX_RETRIES=2 => attempts 1..3
        # ---- HEAD
        try:
            r = session.head(
                url,
                allow_redirects=True,
                timeout=TIMEOUT_SECONDS,
            )
            status = r.status_code
            ct = r.headers.get("Content-Type")
            cl = r.headers.get("Content-Length")
            content_length = int(cl) if cl and cl.isdigit() else None

            ok = status in ALLOWED_STATUS_FOR_TRY
            logs.append(
                ValidationLog(resource_id, url, attempt, "HEAD", ok, status, "")
            )

            # Some servers refuse HEAD or give useless headers.
            # If ok and content-type looks informative, we can accept.
            # Otherwise fallback to GET.
            if ok and ct:
                return True, status, r.url, ct, content_length

            # If HEAD not ok, try GET anyway (many sites block HEAD)
        except Exception as e:
            last_error = str(e)
            logs.append(
                ValidationLog(resource_id, url, attempt, "HEAD", False, None, last_error)
            )

        # ---- GET (stream=True so we don't download full PDFs; we only need headers)
        try:
            r = session.get(
                url,
                allow_redirects=True,
                timeout=TIMEOUT_SECONDS,
                stream=True,
            )
            status = r.status_code
            ct = r.headers.get("Content-Type")
            cl = r.headers.get("Content-Length")
            content_length = int(cl) if cl and cl.isdigit() else None

            ok = status in ALLOWED_STATUS_FOR_TRY
            logs.append(
                ValidationLog(resource_id, url, attempt, "GET", ok, status, "")
            )

            # Close the connection quickly
            try:
                r.close()
            except Exception:
                pass

            if ok:
                return True, status, r.url, ct, content_length

        except Exception as e:
            last_error = str(e)
            logs.append(
                ValidationLog(resource_id, url, attempt, "GET", False, None, last_error)
            )

        time.sleep(SLEEP_BETWEEN_RETRIES_SECONDS)

    return False, None, None, None, None


def decide(ok: bool, detected_type: str, url: str) -> Tuple[str, str]:
    """
    Decide whether we should attempt extraction later.
    Returns: (final_decision, skip_reason)
    """
    if not ok:
        return "skip", "unreachable_or_error"

    # If we can't classify, we can still choose to extract later,
    # but keeping a strict policy improves defensibility.
    if detected_type == "unknown":
        # You can relax this later if needed:
        return "skip", "unknown_content_type"

    # Otherwise we extract
    return "extract", ""


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(
            f"Input not found: {IN_CSV}\n"
            "Make sure you created resources_included_only.csv first."
        )

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV, dtype=object)

    for col in [ID_COL, TITLE_COL, URL_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Columns: {list(df.columns)}")

    # Build HTTP session
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "edi-hubplus-dissertation-bot/1.0 (link validation; contact: student)",
            "Accept": "*/*",
        }
    )

    manifest_rows: List[Dict[str, Any]] = []
    logs: List[ValidationLog] = []

    for _, row in df.iterrows():
        rid = str(row.get(ID_COL, "")).strip()
        title = str(row.get(TITLE_COL, "")).strip()
        url_raw = row.get(URL_COL, "")

        url = normalise_url(url_raw)

        # Empty URL => skip
        if not url:
            manifest_rows.append(
                {
                    "ID": rid,
                    "Title": title,
                    "Link": "",
                    "url_status": "missing",
                    "http_status_code": "",
                    "final_url": "",
                    "content_type": "",
                    "detected_type": "unknown",
                    "content_length": "",
                    "final_decision": "skip",
                    "skip_reason": "missing_url",
                }
            )
            continue

        ok, status_code, final_url, content_type, content_length = request_with_fallback(
            session=session,
            url=url,
            logs=logs,
            resource_id=rid,
        )

        detected = detect_type(final_url or url, content_type)
        final_decision, skip_reason = decide(ok, detected, final_url or url)

        url_status = "ok" if ok else "error"

        manifest_rows.append(
            {
                "ID": rid,
                "Title": title,
                "Link": url,
                "url_status": url_status,
                "http_status_code": status_code if status_code is not None else "",
                "final_url": final_url or "",
                "content_type": content_type or "",
                "detected_type": detected,
                "content_length": content_length if content_length is not None else "",
                "final_decision": final_decision,
                "skip_reason": skip_reason,
            }
        )

    # Write outputs
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(OUT_MANIFEST, index=False)

    log_df = pd.DataFrame([asdict(l) for l in logs])
    log_df.to_csv(OUT_LOG, index=False)

    # Summary prints (helpful for you + methodology)
    print(" Link validation complete")
    print(f"Input rows: {len(df)}")
    print(f"Manifest saved: {OUT_MANIFEST}")
    print(f"Log saved: {OUT_LOG}")
    print("Counts by decision:")
    print(manifest_df["final_decision"].value_counts(dropna=False).to_string())
    print("\nCounts by detected_type:")
    print(manifest_df["detected_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()