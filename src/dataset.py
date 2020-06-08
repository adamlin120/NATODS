from typing import List, Dict, Union

import torch
from torch.utils.data import Dataset

from src.preprocess import TurnState, BatchState
from src.tokenizer import WordLevelTokenizer
from src.ontology import slots

SLOT_TYPES = ['domain', 'slot', 'gate', 'val', 'fertility']


class MultiWozDSTDataset(Dataset):
    slot_names = slots

    max_length = 512
    ignore_idx = -100

    def __init__(self,
                 turns: List[TurnState],
                 tokenizer: WordLevelTokenizer,
                 multiplier: int = 1
                 ):
        self.turns = turns
        self.tokenizer = tokenizer
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(MultiWozDSTDataset.max_length)
        self.multiplier = multiplier

    def __len__(self) -> len:
        return len(self.turns)

    def __getitem__(self, index: int) -> TurnState:
        return self.turns[index]

    def collate_fn(self,
                   examples: List[TurnState]
                   ) -> Dict[str, Union[List[str], BatchState]]:
        batch = {}
        for idx, attr in enumerate(MultiWozDSTDataset.slot_names):
            batch[attr] = BatchState([example.states[idx]
                                      for example in examples])

        # histories
        for attr in ['history', 'delex_history']:
            batch[attr] = [getattr(example, attr)
                           for example in examples]
            batch[f'encoded_{attr}'] = self.tokenizer.encode_batch(batch[attr])
            batch[f'mask_{attr}'] = torch.BoolTensor(
                [enc.attention_mask for enc in batch[f'encoded_{attr}']])
            batch[f'ids_{attr}'] = torch.LongTensor(
                [enc.ids for enc in batch[f'encoded_{attr}']]).T

        # gate, fertility encoder's input: domain, slot pairs
        for pos, input_type in enumerate(['domain', 'slot']):
            batch[f'input_{input_type}'] = \
                [' '.join(
                    domain_slot.split('_')[pos]
                    for domain_slot in self.slot_names
                    for _ in range(self.multiplier)
                )] * len(examples)
            batch[f'encoded_input_{input_type}'] = \
                self.tokenizer.encode_batch(batch[f'input_{input_type}'])
            batch[f'ids_input_{input_type}'] = torch.LongTensor(
                [enc.ids for enc in batch[f'encoded_input_{input_type}']]).T
            assert batch[f'ids_input_{input_type}'].size() == \
                   (self.multiplier * len(self.slot_names), len(examples))

        # slot gate, fertility
        batch['gate'] = torch.LongTensor(
            [batch[attr].gate_index for attr in MultiWozDSTDataset.slot_names])
        batch['fertility'] = torch.LongTensor(
            [batch[attr].fertility for attr in MultiWozDSTDataset.slot_names])

        # value encoder's input: (domain, slot pairs) x fertility
        for pos, input_type in enumerate(['domain', 'slot']):
            batch[f'input-fert_{input_type}'] = \
                [' '.join(
                    domain_slot.split('_')[pos]
                    for i, domain_slot in enumerate(self.slot_names)
                    for _ in range(example[i].fertility)
                )
                    for example in examples
                ]
            batch[f'encoded_input-fert_{input_type}'] = \
                self.tokenizer.encode_batch(batch[f'input-fert_{input_type}'])
            batch[f'ids_input-fert_{input_type}'] = torch.LongTensor(
                [enc.ids for enc in batch[f'encoded_input-fert_{input_type}']]
            ).T

        # slot value
        batch['value_str'] = [
            ' '.join(f'[{state.domain}_{state.slot}] {state.value}'
                     for state in example if state)
            for example in examples
        ]
        batch[f'encoded_value'] = self.tokenizer.encode_batch(
            batch['value_str'])
        batch[f'value'] = torch.LongTensor(
            [enc.ids for enc in batch[f'encoded_value']]).T
        batch[f'value'][batch[f'value']==0] = MultiWozDSTDataset.ignore_idx
        return batch
