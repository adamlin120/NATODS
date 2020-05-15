from typing import List


def tokenizer(text: str) -> List[str]:
    return text.strip().split(' ')
