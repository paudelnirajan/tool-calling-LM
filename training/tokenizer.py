"""
Word-level tokenizer for the tool-calling dataset.

Builds a vocabulary from the training sequences, converts text to
integer IDs, and decodes IDs back to text.
"""

import json
import re
from pathlib import Path


# Special tokens — these get the lowest IDs
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<CALL>", "NO_CALL"]

# Characters that should be their own token (punctuation in tool calls)
PUNCT = set('()=,"')


def _tokenize_text(text):
    """Split text into words and punctuation tokens."""
    tokens = []
    for word in text.split():
        buf = ""
        for ch in word:
            if ch in PUNCT:
                if buf:
                    tokens.append(buf)
                    buf = ""
                tokens.append(ch)
            else:
                buf += ch
        if buf:
            tokens.append(buf)
    return tokens


class Tokenizer:
    """Word-level tokenizer with special tokens for tool-calling."""

    def __init__(self):
        self.token2id = {}
        self.id2token = {}

    @property
    def vocab_size(self):
        return len(self.token2id)

    @property
    def pad_id(self):
        return self.token2id["<PAD>"]

    @property
    def bos_id(self):
        return self.token2id["<BOS>"]

    @property
    def eos_id(self):
        return self.token2id["<EOS>"]

    @property
    def unk_id(self):
        return self.token2id["<UNK>"]

    def build_vocab(self, texts, min_freq=1):
        """
        Build vocabulary from a list of strings.

        Args:
            texts: list of strings (e.g. all input_text + target_text).
            min_freq: drop words that appear fewer than this many times.
        """
        # Count word frequencies
        freq = {}
        for text in texts:
            for tok in _tokenize_text(text):
                freq[tok] = freq.get(tok, 0) + 1

        # Start with special tokens
        self.token2id = {}
        for tok in SPECIAL_TOKENS:
            self.token2id[tok] = len(self.token2id)

        # Add words sorted by frequency (most common first)
        for tok, count in sorted(freq.items(), key=lambda x: -x[1]):
            if count >= min_freq and tok not in self.token2id:
                self.token2id[tok] = len(self.token2id)

        self.id2token = {i: t for t, i in self.token2id.items()}
        return self

    def encode(self, text, add_bos=False, add_eos=False):
        """Convert a string to a list of token IDs."""
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for tok in _tokenize_text(text):
            ids.append(self.token2id.get(tok, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        """Convert a list of token IDs back to a string."""
        tokens = []
        for i in ids:
            tok = self.id2token.get(i, "<UNK>")
            if tok in ("<PAD>", "<BOS>", "<EOS>"):
                continue
            tokens.append(tok)
        # Join with spaces, then clean up spacing around punctuation
        text = " ".join(tokens)
        for p in PUNCT:
            text = text.replace(f" {p} ", p).replace(f" {p}", p).replace(f"{p} ", p)
        return text

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.token2id, f, indent=2)

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path) as f:
            tok.token2id = json.load(f)
        tok.id2token = {int(i): t for t, i in tok.token2id.items()}
        return tok

    @classmethod
    def from_data_files(cls, *paths):
        """Build a tokenizer from one or more sequence JSON files."""
        texts = []
        for p in paths:
            with open(p) as f:
                data = json.load(f)
            for ex in data:
                texts.append(ex["input_text"])
                texts.append(ex["target_text"])
        tok = cls()
        tok.build_vocab(texts)
        return tok


if __name__ == "__main__":
    # Quick test with the actual dataset
    base = Path(__file__).resolve().parent.parent / "data_repo" / "data"
    tok = Tokenizer.from_data_files(
        base / "train_sequences.json",
        base / "val_sequences.json",
    )
    print(f"Vocab size: {tok.vocab_size}")

    sample = "<BOS>\nUser: What's the weather in Denver tomorrow?\nAvailable tools: weather(city,date)\n<CALL>"
    ids = tok.encode(sample)
    print(f"Encoded:    {ids[:20]}...")
    print(f"Decoded:    {tok.decode(ids)}")
