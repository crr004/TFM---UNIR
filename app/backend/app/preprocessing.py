import re
import unicodedata
from html import unescape


def minimal_beto_clean(text: str) -> str:
    normalized = unescape("" if text is None else str(text))
    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = normalized.replace("\u00a0", " ").replace("\u200b", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def build_full_text(title: str, body: str) -> str:
    return minimal_beto_clean(f"{title or ''} {body or ''}")
