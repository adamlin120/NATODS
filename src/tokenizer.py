"""
Tokenizer is mainly from: https://github.com/huggingface/tokenizers/issues/244
"""

import json
import logging
from collections import Counter
from typing import Union, Optional, List
from pathlib import Path

from tokenizers import Tokenizer, AddedToken, pre_tokenizers
from tokenizers.implementations import BaseTokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase, unicode_normalizer_from_str, \
    Sequence

from src.preprocess import TurnState


UTILS_TOKENS = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2}
SYSTEM_BOS = '[SYSTEM]'
USER_BOS = '[USER]'
SLOT_BOS = {
    f'[{d_s}]': len(UTILS_TOKENS) + i
    for i, d_s in enumerate(
        ['attraction_area', 'attraction_name', 'attraction_type',
         'hotel_area', 'hotel_day', 'hotel_internet', 'hotel_name',
         'hotel_parking', 'hotel_people', 'hotel_pricerange',
         'hotel_stars', 'hotel_stay', 'hotel_type',
         'restaurant_area', 'restaurant_day', 'restaurant_food',
         'restaurant_name', 'restaurant_people',
         'restaurant_pricerange', 'restaurant_time',
         'taxi_arriveby', 'taxi_departure', 'taxi_destination',
         'taxi_leaveat',
         'train_arriveby', 'train_day', 'train_departure',
         'train_destination', 'train_leaveat', 'train_people'])
}
SPECIAL_TOKENS = {
    **UTILS_TOKENS,
    **SLOT_BOS
}


class WordLevelTokenizer(BaseTokenizer):
    """ WordLevelTokenizer
    Represents a simple word level tokenization (split on whitespaces).
    """

    def __init__(
            self,
            vocab_file: Optional[str] = None,
            unk_token: Union[str, AddedToken] = "[UNK]",
            pad_token: Union[str, AddedToken] = "[PAD]",
            mask_token: Union[str, AddedToken] = "[MASK]",
            lowercase: bool = False,
            unicode_normalizer: Optional[str] = None,
    ):
        if vocab_file is not None:
            logging.info(f"Initiating tokenizer at {vocab_file}")
            tokenizer = Tokenizer(
                WordLevel(vocab=vocab_file, unk_token=unk_token))
        else:
            tokenizer = Tokenizer(WordLevel(unk_token=unk_token))

        # Let the tokenizer know about special tokens if they are part of the vocab
        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])
        if tokenizer.token_to_id(str(pad_token)) is not None:
            tokenizer.add_special_tokens([str(pad_token)])
        if tokenizer.token_to_id(str(mask_token)) is not None:
            tokenizer.add_special_tokens([str(mask_token)])

        # Check for Unicode normalization first (before everything else)
        normalizers = []

        if unicode_normalizer:
            normalizers += [unicode_normalizer_from_str(unicode_normalizer)]

        if lowercase:
            normalizers += [Lowercase()]

        # Create the normalizer structure
        if len(normalizers) > 0:
            if len(normalizers) > 1:
                tokenizer.normalizer = Sequence(normalizers)
            else:
                tokenizer.normalizer = normalizers[0]

        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

        parameters = {
            "model": "WordLevel",
            "unk_token": unk_token,
            "pad_token": pad_token,
            "mask_token": mask_token,
            "lowercase": lowercase,
            "unicode_normalizer": unicode_normalizer,
        }

        super().__init__(tokenizer, parameters)


def collect_words(turns) -> List[str]:
    attributes = ['history', 'delex_history']
    words = [word
             for attr in attributes
             for turn in turns
             for word in getattr(turn, attr).split()] + \
            [state.value
             for turns in turns
             for state in turns.states]
    return words


def get_tokenizer(turns: List[TurnState],
                  vocab_file: str) -> WordLevelTokenizer:
    logging.info("Collecting words for tokenizer")
    logging.info(f"Special tokens: {SPECIAL_TOKENS}")
    words = collect_words(turns)
    logging.info(f"Collected words: {len(words)}")
    counter = Counter(words)
    logging.info(f"Collected unique words: {len(counter)}")
    vocab = {
        **SPECIAL_TOKENS,
        **{w: i + len(SPECIAL_TOKENS) for i, (w, f) in
           enumerate(counter.most_common())}
    }
    Path(vocab_file).parent.mkdir(exist_ok=True, parents=True)
    json.dump(vocab, open(vocab_file, 'w'))
    logging.info(f"Saving tokenizer at {vocab_file}")
    tokenizer = WordLevelTokenizer(vocab_file)
    return tokenizer
