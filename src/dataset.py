from typing import List, Dict, Union

import ipdb
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocess import preprocess, TurnState, BatchState
from src.tokenizer import WordLevelTokenizer, get_tokenizer

SLOT_TYPES = ['domain', 'slot', 'gate', 'val', 'fertility']


class MultiWozDSTDataset(Dataset):
    num_slots = 30
    slot_names = ['attraction_area', 'attraction_name', 'attraction_type',
                  'hotel_area', 'hotel_day', 'hotel_internet', 'hotel_name',
                  'hotel_parking', 'hotel_people', 'hotel_pricerange',
                  'hotel_stars', 'hotel_stay', 'hotel_type',
                  'restaurant_area', 'restaurant_day', 'restaurant_food',
                  'restaurant_name', 'restaurant_people',
                  'restaurant_pricerange', 'restaurant_time',
                  'taxi_arriveby', 'taxi_departure', 'taxi_destination',
                  'taxi_leaveat',
                  'train_arriveby', 'train_day', 'train_departure',
                  'train_destination', 'train_leaveat', 'train_people']

    max_length = 512

    def __init__(self,
                 turns: List[TurnState],
                 tokenizer: WordLevelTokenizer
                 ):
        self.turns = turns
        self.tokenizer = tokenizer
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(MultiWozDSTDataset.max_length)

    def __len__(self) -> len:
        return len(self.turns)

    def __getitem__(self, index: int) -> TurnState:
        return self.turns[index]

    def collate_fn(self,
                   examples: List[TurnState]
                   ) -> Dict[str, Union[List[str], BatchState]]:
        batch = {}
        for attr in ['history', 'delex_history']:
            batch[attr] = [getattr(example, attr)
                           for example in examples]
            batch[f'encoded_{attr}'] = self.tokenizer.encode_batch(batch[attr])
            batch[f'ids_{attr}'] = torch.LongTensor(
                [enc.ids for enc in batch[f'encoded_{attr}']])
        for idx, attr in zip(range(MultiWozDSTDataset.num_slots),
                             MultiWozDSTDataset.slot_names):
            batch[attr] = BatchState([example.states[idx]
                                      for example in examples])
            batch[f'{attr}_gate'] = torch.LongTensor(batch[attr].gate_index)
            batch[f'{attr}_fertility'] = torch.LongTensor(batch[attr].fertility)
            batch[f'encoded_{attr}_value'] = self.tokenizer.encode_batch(
                batch[attr].value)
            batch[f'ids_{attr}_value'] = torch.LongTensor(
                [enc.ids for enc in batch[f'encoded_{attr}_value']])
        return batch


if __name__ == '__main__':
    train_path = './.data/MultiWoz_2.1_NADST_Version/data2.1/nadst_train_dials.json'
    ontology_path = './.data/MultiWoz_2.1_NADST_Version/data2.1/multi-woz/MULTIWOZ2.1/ontology.json'
    vocab_file = 'vocab_file.json'
    train_turns = preprocess(train_path, ontology_path)
    tokenizer = get_tokenizer(train_turns, vocab_file)
    train = MultiWozDSTDataset(train_turns, tokenizer)
    for batch in DataLoader(train, 2, collate_fn=train.collate_fn):
        print(batch)
        ipdb.set_trace()
