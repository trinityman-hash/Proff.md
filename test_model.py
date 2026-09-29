import json
from drle.model import DRLEModel
from drle.tokenizer import encode, decode

def test_tokenizer_deterministic():
    assert encode("Hello, world!") == ["hello", ",", "world", "!"]
    assert decode(["hello", ",", "world", "!"]) == "hello, world!"

def test_context_order_matters():
    m=DRLEModel(3); m.add_text("dog bites man"); m.add_text("man bites dog")
    assert m.predict("dog bites").token == "man"
    assert m.predict("man bites").token == "dog"

def test_repeatable_and_stable_tie_break():
    m=DRLEModel(2); m.add_text("a b"); m.add_text("a c")
    assert m.predict("a").token == m.predict("a").token

def test_save_load_roundtrip(tmp_path):
    m=DRLEModel(3); m.add_text("the river flows"); path=tmp_path/"model.json"; m.save(path)
    n=DRLEModel.load(path)
    assert n.predict("the river").token == m.predict("the river").token
    assert n.token_count == m.token_count

def test_unknown_context_halts():
    m=DRLEModel(2); m.add_text("alpha beta")
    assert m.predict("unseen context").token is None

def test_plain_text_corpus_reader(tmp_path):
    from drle.corpus import read_corpus
    path = tmp_path / "seed.txt"
    path.write_text("first sequence\n\nsecond sequence\n", encoding="utf-8")
    assert list(read_corpus(path)) == ["first sequence", "second sequence"]
