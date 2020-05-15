from abc import ABC
from dataclasses import dataclass, asdict
from typing import List, Tuple

from src.data.multiwoz.attraction import MultiWozAttraction
from src.data.multiwoz.hotel import MultiWozHotel
from src.data.multiwoz.restaurant import MultiWozRestaurant
from src.data.multiwoz.taxi import MultiWozTaxi
from src.data.multiwoz.train import MultiWozTrain
from src.data.slot import DomainSlotValue

DST_DOMAIN = ['hotel', 'train', 'attraction', 'restaurant', 'taxi']


@dataclass
class MultiWozBookSlots(ABC):
    booked: List


@dataclass
class MultiWozSemiSlots(ABC):
    _domain: str

    @property
    def domain(self):
        return self._domain

    @property
    def sorted_items(self) -> List[Tuple[str, str]]:
        return [
            (field_name, field_value)
            for field_name, field_value in sorted(asdict(self).items())
            if not field_name.startswith('_')
        ]

    @property
    def domain_slot_value(self) -> List[DomainSlotValue]:
        return [
            DomainSlotValue(self.domain, field_name, field_value)
            for field_name, field_value in self.sorted_items
        ]

    @property
    def is_empty(self) -> bool:
        return all(pair.is_empty for pair in self.domain_slot_value)

    def __str__(self):
        return ' '.join(map(str, self.domain_slot_value)).strip()


@dataclass
class MultiWozDomainState(ABC):
    book: MultiWozBookSlots
    semi: MultiWozSemiSlots

    @property
    def is_empty(self):
        return not self.semi or self.semi.is_empty

    def __str__(self) -> str:
        return str(self.semi)


@dataclass
class MultiWozDialogueState:
    hotel: MultiWozHotel
    attraction: MultiWozAttraction
    restaurant: MultiWozRestaurant
    train: MultiWozTrain
    taxi: MultiWozTaxi

    @classmethod
    def domain_slots(cls) -> List[Tuple[str, str]]:
        pairs = []
        for domain, field_type in cls.__annotations__.items():
            domain_state_class = field_type.__args__[0]
            semi_state_class = domain_state_class.__annotations__.get('semi',
                                                                      None)
            if semi_state_class is None:
                continue
            for slot_name in semi_state_class.__dataclass_fields__.keys():
                pairs.append((domain, slot_name))
        return pairs

    @property
    def sorted_states(self) -> List[MultiWozDomainState]:
        return [getattr(self, key)
                for key, _ in sorted(asdict(self).items())
                if not key.startswith('_')]

    @property
    def non_empty_sorted_states(self) -> List[MultiWozDomainState]:
        return [state for state in self.sorted_states if not state.is_empty]

    def __str__(self) -> str:
        return ' '.join(map(str, self.non_empty_sorted_states)).strip()
