"""
Tokenizer for the SmSA dataset.

The SmSA text is already pre-tokenized with whitespace separating words
and punctuation (e.g. "tahu ." not "tahu."). So tokenization here is
intentionally simple: lowercase + split on whitespace. No subword/BPE
needed for this project (that's reserved for Project 8 - IndoBERT).
"""

from typing import List


def tokenize(text: str) -> List[str]:
    """Split pre-tokenized SmSA text into a list of lowercase tokens.

    Args:
        text: raw text from a SmSA row, e.g. "warung ini dimiliki ."

    Returns:
        List of tokens, e.g. ["warung", "ini", "dimiliki", "."]
    """
    if not text:
        return []
    return text.lower().strip().split()


if __name__ == "__main__":
    sample = "Warung ini dimiliki oleh pengusaha pabrik tahu ."
    print(tokenize(sample))