import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from src.ontology import get_domain_slot_pairs


@dataclass
class State:
    domain: str
    slot: str
    value: str

    gate2index: Dict[str, int] = field(default_factory=lambda: {
        'none': 0,
        'dontcare': 1,
        'gen': 2
    })

    @property
    def gate(self) -> str:
        if self.value in {'none'}:
            return 'none'
        elif self.value in {'dontcare'}:
            return 'dontcare'
        else:
            return 'gen'

    @property
    def gate_index(self) -> int:
        return self.gate2index[self.gate]

    @property
    def fertility(self) -> int:
        if self:
            return len(self.value.split())
        else:
            return 0

    def __bool__(self):
        return self.gate != 'none'

    def __str__(self):
        return '\t'.join([
            self.domain,
            self.slot,
            self.gate,
            self.value,
            str(self.fertility),
        ]) + '\n'


@dataclass
class TurnState:
    states: List[State]
    history: str
    delex_history: str

    def __str__(self):
        return self.history + '\n' + \
               self.delex_history + '\n' + \
               ''.join(map(str, self.states)) + '\n'

    def __getitem__(self, index: int) -> State:
        return self.states[index]


@dataclass
class BatchState:
    states: List[State]

    def __getattr__(self, item: str):
        assert all(s.domain == self.states[0].domain for s in self.states)
        assert all(s.slot == self.states[0].slot for s in self.states)
        return [getattr(s, item) for s in self.states]


def add_dialogue_index(dialogue: Dict) -> Dict:
    for turn in dialogue['dialogue']:
        turn['dialogue_idx'] = dialogue['dialogue_idx']
    assert all('dialogue_idx' in turn for turn in dialogue['dialogue'])
    return dialogue


def add_history_transcript(dialogue: Dict) -> Dict:
    assert [turn['turn_idx'] for turn in dialogue['dialogue']] \
           == list(range(len(dialogue['dialogue'])))

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
        turn.update(deepcopy(history))
    assert all(all(k in turn for k in history.keys())
               for turn in dialogue['dialogue'])
    return dialogue


def get_all_turns(
    data: List[Dict],
    ontology: List[Tuple[str, str]],
) -> List[TurnState]:
    turns = []
    for dial in data:
        for turn in dial['dialogue']:
            states = []
            turn_state: Dict[Tuple[str, str], str] = {}
            for belief_state in turn['belief_state']:
                domain_slot, value = belief_state['slots'][0]
                value = value.strip()
                domain, slot = _parse_domain_slot(domain_slot)
                turn_state[(domain, slot)] = value
            for domain, slot in ontology:
                value = turn_state.get((domain, slot), 'none')
                states.append(State(domain, slot, value))
            assert len(states) == 30
            history = _concat_transcript(
                    turn['history_transcript'],
                    turn['history_system_transcript'])
            delex_history = _concat_transcript(
                    turn['history_delex_transcript'], 
                    turn['history_delex_system_transcript'])
            turns.append(TurnState(states, history, delex_history))
    return turns


def _parse_domain_slot(domain_slot: str):
    domain, slot = domain_slot.split('-')
    domain = domain.strip()
    slot = slot.replace(' ', '').strip()
    return domain, slot


def _concat_transcript(
    user_transcript: List[str], 
    system_transcript: List[str]
) -> str:
    assert len(user_transcript) == len(system_transcript)
    history = ''
    for user, sys in zip(user_transcript, system_transcript):
        sys = sys.strip()
        user = user.strip()
        if sys:
            history += " [SYSTEM] " + sys
        history += " [USER] " + user
    return history.strip()


def preprocess(
        multiwoz_path: str,
        ontology_path: str,
        output_path: Optional[str] = None,
) -> List[TurnState]:
    with open(multiwoz_path) as f:
        data: List[Dict] = json.load(f)
    ontology = get_domain_slot_pairs(ontology_path)
    data = [add_dialogue_index(dial) for dial in data]
    data = [add_history_transcript(dial) for dial in data]
    turns = get_all_turns(data, ontology)
    if output_path:
        Path(output_path).write_text(''.join(map(str, turns)))
    return turns
