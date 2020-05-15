import logging
from dataclasses import asdict
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from torchtext.data import ReversibleField
from tqdm.auto import tqdm

from src.data.multiwoz.state import MultiWozDialogueState

logger = logging.getLogger(__name__)

USR_BOS = '<USR-BOS>'
SYS_BOS = '<SYS-BOS>'

source_repeat_time = 4


class MultiWozDataset(Dataset):
    _source: List[Tuple[str, str]] = MultiWozDialogueState.domain_slots()

    def __init__(
            self,
            data: Dict[str, List],
    ) -> None:
        self._data = data
        self._instances: List[Dict] = MultiWozDataset._make_instance(self._data)

        self.text_field: ReversibleField = None

    def __len__(self) -> int:
        return len(self._instances)

    def __getitem__(self, index: int) -> Dict:
        instance = self._instances[index]
        context = self._clean_context(instance['context'])
        states = instance['belief_state']
        target = str(states)
        return {
            'target': target,
            'context': context,
        }

    def collate_fn(
            self,
            batch: List[Dict[str, str]],
    ) -> Dict[str, torch.Tensor]:
        bs = len(batch)
        domains = torch.LongTensor([self.text_field.vocab.stoi[domain]
                                    for domain, slot in self.source]
                                   ).repeat(bs, 1)
        slots = torch.LongTensor([self.text_field.vocab.stoi[slot]
                                  for domain, slot in self.source]
                                 ).repeat(bs, 1)
        source_lengths = torch.LongTensor([len(self.source)] * bs)
        targets, target_lengths = self.text_field.process(
            [self.text_field.tokenize(ins['target']) for ins in batch])
        contexts, context_lengths = self.text_field.process(
            [self.text_field.tokenize(ins['context']) for ins in batch])
        collated_batch = {
            'domain': domains,
            'slot': slots,
            'target': targets,
            'source_lengths': source_lengths,
            'target_lengths': target_lengths,
            'context': contexts,
        }
        return collated_batch

    @property
    def source(self):
        return [ds
                for ds in self._source
                for _ in range(source_repeat_time)]

    @staticmethod
    def _clean_context(context: List[str]) -> str:
        context = [
            MultiWozDataset._pad_bos_eos(
                utterance.strip(),
                SYS_BOS if i % 2 else USR_BOS,
                SYS_EOS if i % 2 else USR_EOS
            )
            for i, utterance in enumerate(context)
        ]
        context = MultiWozDataset._concat_strings(context)
        return context

    @staticmethod
    def _pad_bos_eos(utterance: str, bos: str, eos: str):
        assert bos in {USR_BOS, SYS_BOS}
        assert eos in {USR_EOS, SYS_EOS}
        return ' '.join([bos, utterance, eos])

    @staticmethod
    def _concat_strings(utterances: List[str]) -> str:
        return ' '.join(utterances)

    @staticmethod
    def _make_instance(dataset: Dict[str, List]) -> List[Dict]:
        instance_keys = ['utterance', 'dialog_act', 'context',
                         'context_dialog_act', 'belief_state',
                         'last_opponent_utterance', 'last_self_utterance',
                         'session_id', 'span_info', 'terminated', 'goal']
        assert all(len(dataset[key]) == len(dataset[instance_keys[0]])
                   for key in instance_keys)
        instances: List[Dict] = [dict(zip(dataset, t))
                                 for t in zip(*dataset.values())]
        assert len(instances) == len(dataset[instance_keys[0]])

        for prev_turn, curr_turn in tqdm(zip([{}] + instances[:-1], instances),
                                         total=len(instances)):
            are_in_same_dialogue = \
                prev_turn and \
                prev_turn['session_id'] == curr_turn['session_id'] and \
                2 + len(prev_turn['context']) == len(curr_turn['context'])

            config = Config(check_types=False)
            curr_turn['prev_states'] = from_dict(
                MultiWozDialogueState,
                asdict(prev_turn['belief_state']) if are_in_same_dialogue
                else {},
                config
            )
            curr_turn['belief_state'] = from_dict(
                MultiWozDialogueState,
                curr_turn['belief_state'],
                config
            )
        assert all(
            key in single_instance and
            len(single_instance) == len(instance_keys) + 1
            for single_instance in instances
            for key in instance_keys
        )
        return instances
