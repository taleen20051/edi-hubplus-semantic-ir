from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import requests


# Input file containing resources that were marked as included in the spreadsheet.
IN_CSV = Path("data/processed/resources_included_only.csv")
# Summary output recording the final validation decision for each resource.
OUT_MANIFEST = Path("data/processed/resource_manifest.csv")
# Detailed log of each HTTP attempt made during validation.
OUT_LOG = Path("data/processed/link_validation_log.csv")

ID_COL = "ID"
TITLE_COL = "Title"
URL_COL = "Link"

# Network settings used for reproducible and controlled validation.
TIMEOUT_SECONDS = 15
MAX_RETRIES = 2
SLEEP_BETWEEN_RETRIES_SECONDS = 1.0

# HTTP statuses treated as accessible enough to attempt later extraction.
# Some servers block HEAD requests, so GET fallback is still used where needed.
ALLOWED_STATUS_FOR_TRY = {200, 201, 202, 203, 204, 206, 301, 302, 303, 307, 308}

# Used when the server content-type is missing or unreliable.
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
    """Clean simple spreadsheet URL artefacts without changing the URL meaning."""
    u = str(url).strip()
    # Remove line breaks that can appear when links are copied from Excel.
    u = u.replace("\n", "").replace("\r", "")
    return u


def detect_type(url: str, content_type: Optional[str]) -> str:
    """Classify a resource as 'pdf', 'html', or 'unknown'."""
    ct = (content_type or "").lower().strip()

    # Prefer server-provided content-type when available.
    if "application/pdf" in ct:
        return "pdf"
    if "text/html" in ct:
        return "html"

    # Fall back to URL extension and general text content types.
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
    Validate a URL using HEAD first, then GET if HEAD is blocked or incomplete.
    Returns: (ok, status_code, final_url, content_type, content_length)
    """
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 2):  # e.g. MAX_RETRIES=2 gives 3 attempts.
        # First try HEAD to inspect the resource without downloading the body.
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

            # Accept HEAD only when it succeeds and gives useful type information.
            if ok and ct:
                return True, status, r.url, ct, content_length

        except Exception as e:
            last_error = str(e)
            logs.append(
                ValidationLog(resource_id, url, attempt, "HEAD", False, None, last_error)
            )

        # Fall back to GET because many websites block or mishandle HEAD requests.
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

            # Close streamed response after reading headers.
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

        # Brief pause before retrying to avoid aggressive repeated requests.
        time.sleep(SLEEP_BETWEEN_RETRIES_SECONDS)

    return False, None, None, None, None


def decide(ok: bool, detected_type: str, url: str) -> Tuple[str, str]:
    """
    Decide whether a validated resource should be passed to text extraction.
    Returns: (final_decision, skip_reason)
    """
    if not ok:
        return "skip", "unreachable_or_error"

    # Keep extraction strict by only passing clearly supported resource types.
    if detected_type == "unknown":
        return "skip", "unknown_content_type"

    return "extract", ""


def main() -> None:
    # This stage depends on the included-only dataset from the filtering step.
    if not IN_CSV.exists():
        raise FileNotFoundError(
            f"Input not found: {IN_CSV}\n"
            "Make sure you created resources_included_only.csv first."
        )

    # Create output directories if they do not already exist.
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)

    # Read the included resources while preserving spreadsheet values.
    df = pd.read_csv(IN_CSV, dtype=object)

    # Validate the minimum schema needed for URL checking.
    for col in [ID_COL, TITLE_COL, URL_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Columns: {list(df.columns)}")

    # Reuse a session for efficiency and identify the request purpose politely.
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

        # Record missing URLs explicitly so they remain visible in the manifest.
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

        # Validate the URL and collect final redirected URL/type metadata.
        ok, status_code, final_url, content_type, content_length = request_with_fallback(
            session=session,
            url=url,
            logs=logs,
            resource_id=rid,
        )

        detected = detect_type(final_url or url, content_type)
        final_decision, skip_reason = decide(ok, detected, final_url or url)

        url_status = "ok" if ok else "error"

        # Store one manifest row per input resource for traceability.
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

    # Save the summary manifest and detailed request log.
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(OUT_MANIFEST, index=False)

    log_df = pd.DataFrame([asdict(l) for l in logs])
    log_df.to_csv(OUT_LOG, index=False)

    # Print a concise summary for reproducibility notes and report checks.
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