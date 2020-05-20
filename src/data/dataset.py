import json
from typing import List, Dict

from torch.utils.data import Dataset


def add_dialogue_index(dialogue: Dict) -> Dict:
    for turn in dialogue['dialogue']:
        turn['dialogue_idx'] = dialogue['dialogue_idx']
    assert all('dialogue_idx' in turn for turn in dialogue['dialogue'])
    return dialogue


def add_history_transcript(dialogue: Dict) -> Dict:
    history = {
        'history_system_transcript': [],
        'history_delex_system_transcript': [],
        'history_transcript': [],
        'history_delex_transcript': [],
    }
    for turn in dialogue['dialogue']:
        for k, v in history.items():
            turn_key = k.split('_', 1)[1]
            history[k].append(turn[turn_key])
        turn.update(history)
    assert all(all(k in turn for k in history.keys())
               for turn in dialogue['dialogue'])
    return dialogue


def get_all_turns(data: List[Dict]) -> List[Dict]:
    return [turn for dial in data for turn in dial['dialogue']]


class MultiWozDataset(Dataset):
    def __init__(self,
                 path: str,
                 ):
        self.path = path
        with open(self.path) as f:
            self.data: List[Dict] = json.load(f)
        self.data = [add_dialogue_index(dial) for dial in self.data]
        self.data = [add_history_transcript(dial) for dial in self.data]
        self.turns = get_all_turns(self.data)

    def __len__(self):
        return len(self.turns)

    def __getitem__(self, index: int):
        return self.turns[index]

    @staticmethod
    def collate(batch: List[Dict]):
        pass

