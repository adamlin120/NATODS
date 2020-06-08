import json
from typing import Dict, List, Tuple


attraction_slots = [
    'attraction_area', 'attraction_name', 'attraction_type',
]
hotel_slots = [
    'hotel_area', 'hotel_day', 'hotel_internet', 'hotel_name', 'hotel_parking',
    'hotel_people', 'hotel_pricerange', 'hotel_stars', 'hotel_stay',
    'hotel_type',
]
restaurant_slots = [
    'restaurant_area', 'restaurant_day', 'restaurant_food', 'restaurant_name',
    'restaurant_people', 'restaurant_pricerange', 'restaurant_time',
]
taxi_slots = [
    'taxi_arriveby', 'taxi_departure', 'taxi_destination', 'taxi_leaveat',
]
train_slots = [
    'train_arriveby', 'train_day', 'train_departure', 'train_destination',
    'train_leaveat', 'train_people'
]
slots = attraction_slots + \
        hotel_slots + \
        restaurant_slots + \
        taxi_slots + \
        train_slots


def _normalize(name: str) -> str:
    return name.replace(' ', '').lower().strip()


def get_domain_slot_pairs(ontology_path: str) -> List[Tuple[str, str]]:
    """
    1. get domain slot names from ontology file
    2. split domain slot apart by '-'
    3. remove keyword "book" in slot name
    e.g. "book stay" -> "stay"
    4. remove "bus" and "hospital" domain

    Args:
        ontology_path: path to multiwoz ontology.json


    Returns:
        List of (domain, slot) pairs

    """
    with open(ontology_path) as f:
        ontology: Dict[str, List[str]] = json.load(f)
    domain_slot_pairs = [k.split('-') for k in ontology.keys()]
    # remove "bus" & "hospital" domains
    domain_slot_pairs = [(d, s)
                         for d, s in domain_slot_pairs
                         if d not in {'bus', 'hospital'}]
    # filter out "book"
    domain_slot_pairs = [
        (_normalize(domain), _normalize(slot.replace('book', '')))
        for domain, slot in domain_slot_pairs
    ]
    domain_slot_pairs = sorted(domain_slot_pairs)
    assert len(domain_slot_pairs) == 30
    return domain_slot_pairs
