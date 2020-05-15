from torchtext.data import ReversibleField

from src.data.tokenizer import tokenizer

TEXT_FIELD = ReversibleField(
    sequential=True,
    use_vocab=True,
    tokenize=tokenizer,
    lower=True,
    include_lengths=True,
)
