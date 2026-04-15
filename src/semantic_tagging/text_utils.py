import re

_WS = re.compile(r"\s+")


def normalise_text(s: str) -> str:
    s = (s or "").strip()
    s = _WS.sub(" ", s)
    return s


def make_resource_text(title: str, extracted_text: str, max_chars: int = 8000) -> str:
    """
    Make a stable text representation. We prepend the title to help short docs.
    We cap length so one very long resource doesn't dominate runtime.
    """
    title = normalise_text(title)
    body = normalise_text(extracted_text)
    merged = f"{title}. {body}".strip()
    return merged[:max_chars]