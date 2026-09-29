from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from .corpus import read_corpus
from .model import DRLEModel

def train(input_path: str, output_path: str, max_order: int=5, max_examples: int=0) -> dict:
    model=DRLEModel(max_order=max_order); digest=hashlib.sha256(); examples=0
    for text in read_corpus(input_path):
        digest.update(text.encode("utf-8")); digest.update(b"\n")
        model.add_text(text); examples+=1
        if max_examples and examples>=max_examples: break
    model.save(output_path)
    size=Path(output_path).stat().st_size
    report={"model":str(output_path),"examples":examples,"token_count":model.token_count,"sequence_count":model.sequence_count,"unique_contexts":sum(len(x) for x in model.counts.values()),"unique_transitions":sum(len(c) for order in model.counts.values() for c in order.values()),"model_bytes":size,"corpus_sha256":digest.hexdigest(),"max_order":max_order}
    Path(str(output_path)+".report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",default="artifacts/drle.json"); p.add_argument("--max-order",type=int,default=5); p.add_argument("--max-examples",type=int,default=0)
    a=p.parse_args(); print(json.dumps(train(a.input,a.output,a.max_order,a.max_examples),indent=2))
if __name__=="__main__": main()
