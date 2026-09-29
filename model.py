"""Sparse variable-order count model with deterministic integer scoring."""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from .tokenizer import encode, decode, TOKENIZER_VERSION

FORMAT_VERSION = 1

@dataclass
class Prediction:
    token: str | None
    candidates: list[tuple[str, int]]
    used_orders: list[int]
    reason: str

class DRLEModel:
    def __init__(self, max_order: int = 5):
        if max_order < 1 or max_order > 12: raise ValueError("max_order must be 1..12")
        self.max_order = max_order
        self.counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {k: defaultdict(Counter) for k in range(1, max_order + 1)}
        self.token_count = 0
        self.sequence_count = 0

    def add_text(self, text: str) -> None:
        toks = encode(text)
        if not toks: return
        self.sequence_count += 1
        self.token_count += len(toks)
        padded = ["<s>"] * self.max_order + toks + ["</s>"]
        for i in range(self.max_order, len(padded)):
            target = padded[i]
            for k in range(1, self.max_order + 1):
                ctx = tuple(padded[i-k:i])
                self.counts[k][ctx][target] += 1

    def predict(self, context: str, min_count: int = 1, top_k: int = 8, backoff: bool = True) -> Prediction:
        toks = encode(context)
        gathered: Counter[str] = Counter(); orders=[]
        upper = min(self.max_order, len(toks) if toks else 1)
        for k in range(upper, 0, -1):
            ctx = tuple(toks[-k:]) if toks else tuple(["<s>"] * k)
            found = self.counts[k].get(ctx)
            if found:
                orders.append(k)
                for token, count in found.items():
                    # Longer contexts receive a deterministic integer multiplier.
                    gathered[token] += count * (k * k)
                if not backoff: break
        filtered = [(t,s) for t,s in gathered.items() if s >= min_count and t not in {"<s>"}]
        filtered.sort(key=lambda x: (-x[1], x[0]))
        if not filtered: return Prediction(None, [], orders, "NO_SUPPORTED_CONTINUATION")
        return Prediction(filtered[0][0], filtered[:max(1, top_k)], orders, "OK")

    def generate(self, prompt: str, max_new_tokens: int = 40, min_count: int = 1) -> str:
        if max_new_tokens < 0: raise ValueError("max_new_tokens must be nonnegative")
        output = encode(prompt)
        for _ in range(max_new_tokens):
            pred = self.predict(decode(output), min_count=min_count, top_k=1)
            if pred.token is None or pred.token == "</s>": break
            output.append(pred.token)
        return decode(output)

    def save(self, path: str | Path) -> None:
        p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        rows=[]
        for k in range(1,self.max_order+1):
            for ctx, counter in self.counts[k].items():
                for token,count in counter.items(): rows.append([k,list(ctx),token,count])
        rows.sort(key=lambda r:(r[0],r[1],r[2]))
        payload={"format_version":FORMAT_VERSION,"tokenizer":TOKENIZER_VERSION,"max_order":self.max_order,"token_count":self.token_count,"sequence_count":self.sequence_count,"transitions":rows}
        p.write_text(json.dumps(payload,ensure_ascii=False,separators=(",",":")),encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DRLEModel":
        data=json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format_version") != FORMAT_VERSION: raise ValueError("Unsupported model format")
        if data.get("tokenizer") != TOKENIZER_VERSION: raise ValueError("Tokenizer version mismatch")
        m=cls(data["max_order"]); m.token_count=data["token_count"]; m.sequence_count=data["sequence_count"]
        for k,ctx,token,count in data["transitions"]: m.counts[k][tuple(ctx)][token]=count
        return m
