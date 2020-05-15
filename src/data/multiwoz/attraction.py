from dataclasses import dataclass
from typing import List, Optional

from src.data.multiwoz.state import MultiWozBookSlots, MultiWozSemiSlots, \
    MultiWozDomainState


@dataclass
class MultiWozTrainBookItem:
    trainID: str
    reference: str


@dataclass
class MultiWozAttractionBook(MultiWozBookSlots):
    booked: List[str]


@dataclass
class MultiWozAttractionSemi(MultiWozSemiSlots):
    _domain = 'attraction'
    area: str
    name: str
    type: str


@dataclass
class MultiWozTrainSemi(MultiWozSemiSlots):
    _domain = 'train'
    arriveBy: str
    day: str
    departure: str
    destination: str
    leaveAt: str


@dataclass
class MultiWozTrainBook(MultiWozBookSlots):
    booked: List[MultiWozTrainBookItem]
    people: Optional[str]


@dataclass
class MultiWozAttraction(MultiWozDomainState):
    book: MultiWozAttractionBook
    semi: MultiWozAttractionSemi
