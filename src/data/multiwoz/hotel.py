from dataclasses import dataclass
from typing import List, Optional

from src.data.multiwoz.state import MultiWozBookSlots, MultiWozSemiSlots, \
    MultiWozDomainState


@dataclass
class MultiWozHotelSemi(MultiWozSemiSlots):
    _domain = 'hotel'
    area: str
    internet: str
    name: str
    parking: str
    pricerange: str
    stars: str
    type: str


@dataclass
class MultiWozHotelBookItem:
    name: Optional[str]
    reference: str


@dataclass
class MultiWozHotelBook(MultiWozBookSlots):
    booked: List[MultiWozHotelBookItem]
    people: str
    day: str
    stay: str


@dataclass
class MultiWozHotel(MultiWozDomainState):
    book: MultiWozHotelBook
    semi: MultiWozHotelSemi
