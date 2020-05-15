from dataclasses import dataclass
from typing import List

from src.data.multiwoz.state import MultiWozBookSlots, MultiWozSemiSlots, \
    MultiWozDomainState


@dataclass
class MultiWozTaxiBookItem:
    phone: str
    type: str


@dataclass
class MultiWozTaxiBook(MultiWozBookSlots):
    booked: List[MultiWozTaxiBookItem]


@dataclass
class MultiWozTaxiSemi(MultiWozSemiSlots):
    _domain = 'taxi'
    arriveBy: str
    departure: str
    destination: str
    leaveAt: str


@dataclass
class MultiWozTaxi(MultiWozDomainState):
    book: MultiWozTaxiBook
    semi: MultiWozTaxiSemi
