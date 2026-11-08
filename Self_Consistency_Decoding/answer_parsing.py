"""Answer normalization for voting; no model dependencies."""
import re
from fractions import Fraction

ANSWER_IS = re.compile(r"The answer is\s*", re.IGNORECASE)
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s*/\s*[+-]?\d+(?:\.\d*)?)?")


def parse_arithmetic_first(text: str) -> str | None:
    match = ANSWER_IS.search(text)
    if not match:
        return None
    tail = text[match.end():].strip().replace(",", "").lstrip("$").strip()
    number = NUMBER.match(tail)
    if not number:
        return None
    try:
        parts = number.group().split("/")
        value = Fraction(parts[0].strip())
        if len(parts) == 2:
            value /= Fraction(parts[1].strip())
        return str(value)
    except (ValueError, ZeroDivisionError):
        return None


def parse_commonsense_first(text: str) -> str | None:
    match = ANSWER_IS.search(text)
    if not match:
        return None
    tail = text[match.end():].strip()
    if not tail:
        return None
    return " ".join(tail.splitlines()[0].rstrip(".").casefold().split()) or None
