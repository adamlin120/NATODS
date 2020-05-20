import json
from typing import Dict, List, Tuple


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
        List of all (domain, slot) pairs

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
