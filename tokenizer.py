"""Small deterministic baseline tokenizer; deliberately versioned and replaceable."""
import re

TOKENIZER_VERSION = "drle-regex-en-v1"
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)

def encode(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    return [m.group(0).lower().replace("’", "'") for m in _TOKEN_RE.finditer(text)]

def decode(tokens: list[str]) -> str:
    out = ""
    no_space_before = set(".,!?;:%)]}»”’")
    no_space_after = set("([{«“")
    for tok in tokens:
        if not out:
            out = tok
        elif tok in no_space_before or out[-1] in no_space_after:
            out += tok
        else:
            out += " " + tok
    return out
