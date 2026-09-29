import argparse
from .model import DRLEModel

def main():
    p=argparse.ArgumentParser(prog="drle"); sub=p.add_subparsers(dest="cmd",required=True)
    q=sub.add_parser("predict"); q.add_argument("model"); q.add_argument("context"); q.add_argument("--top-k",type=int,default=8)
    g=sub.add_parser("generate"); g.add_argument("model"); g.add_argument("prompt"); g.add_argument("--tokens",type=int,default=40)
    a=p.parse_args(); m=DRLEModel.load(a.model)
    if a.cmd=="predict":
        r=m.predict(a.context,top_k=a.top_k); print({"token":r.token,"candidates":r.candidates,"orders":r.used_orders,"status":r.reason})
    else: print(m.generate(a.prompt,max_new_tokens=a.tokens))
if __name__=="__main__": main()
