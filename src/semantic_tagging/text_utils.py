import re

# Regular expression used to collapse repeated whitespace
_WS = re.compile(r"\s+")

# Clean text into a stable format for embedding models.
def normalise_text(s: str) -> str:
    s = (s or "").strip()
    s = _WS.sub(" ", s)
    return s


# Build the final text used to embed each resource.
def make_resource_text(title: str, extracted_text: str, max_chars: int = 8000) -> str:
    title = normalise_text(title)
    body = normalise_text(extracted_text)

    # Combine title and extracted body text.
    merged = f"{title}. {body}".strip()

    # Truncate to a fixed maximum size for efficiency.
    return merged[:max_chars]