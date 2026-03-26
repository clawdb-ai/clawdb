from __future__ import annotations


def normalize_text(value: object) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n")


def utf8_text_size(value: object) -> int:
    return len(normalize_text(value).encode("utf-8"))
