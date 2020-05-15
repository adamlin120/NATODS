from dataclasses import dataclass
from typing import List, Optional

from src.data.multiwoz.state import MultiWozBookSlots, MultiWozSemiSlots, \
    MultiWozDomainState


@dataclass
class MultiWozRestaurantBookItem:
    name: Optional[str]
    reference: str


@dataclass
class MultiWozRestaurantSemi(MultiWozSemiSlots):
    _domain = 'restaurant'
    area: str
    food: str
    name: str
    pricerange: str


@dataclass
class MultiWozRestaurantBook(MultiWozBookSlots):
    booked: List[MultiWozRestaurantBookItem]
    day: str
    people: str
    time: str


@dataclass
class MultiWozRestaurant(MultiWozDomainState):
    book: MultiWozRestaurantBook
    semi: MultiWozRestaurantSemi
