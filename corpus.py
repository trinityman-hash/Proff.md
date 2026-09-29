"""Corpus readers for JSONL and Hugging Face datasets."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator


def _text_from_row(row: dict) -> Iterator[str]:
    prompt = row.get("prompt", "")
    messages = row.get("messages", []) or []
    if isinstance(messages, str):
        try: messages = json.loads(messages)
        except json.JSONDecodeError: messages = []
    prefix = str(prompt).strip()
    history: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role, content = msg.get("role", ""), msg.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "assistant":
            context = "\n".join(x for x in [prefix, *history, "assistant:"] if x)
            yield context + " " + content.strip()
        elif role in {"user", "system"}:
            history.append(f"{role}: {content.strip()}")
    if not messages and prefix:
        yield prefix


def read_jsonl(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip(): continue
            try: row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
            if isinstance(row, dict): yield from _text_from_row(row)


def read_corpus(path: str | Path) -> Iterator[str]:
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".txt":
        # One independent training sequence per non-empty line.
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    yield text
        return
    if p.is_file() and p.suffix.lower() in {".jsonl", ".ndjson"}:
        yield from read_jsonl(p); return
    if p.is_file() and p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            if isinstance(row, dict): yield from _text_from_row(row)
        return
    if p.is_dir():
        files = sorted([*p.rglob("*.jsonl"), *p.rglob("*.ndjson"), *p.rglob("*.json")])
        if files:
            for f in files: yield from read_corpus(f)
            return
        parquet = sorted(p.rglob("*.parquet"))
        if parquet:
            try:
                import pyarrow.parquet as pq
            except ImportError as e:
                raise RuntimeError("Install DRLE with: pip install -e '.[hf]'") from e
            for f in parquet:
                table = pq.read_table(f)
                for row in table.to_pylist():
                    if isinstance(row, dict): yield from _text_from_row(row)
            return
    raise FileNotFoundError(f"No supported JSON/JSONL/Parquet corpus found at {p}")
