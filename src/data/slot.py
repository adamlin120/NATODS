from dataclasses import dataclass


@dataclass
class DomainSlotValue:
    _domain: str
    _slot: str
    _value: str

    @property
    def domain(self) -> str:
        return f'<{self._domain.strip()}>'

    @property
    def slot(self) -> str:
        return f'<{self._slot.strip()}>'

    @property
    def value(self) -> str:
        return f'{self._value.strip()}'

    @property
    def gate(self) -> str:
        none_mentions = {
            ''
        }
        dontcare_mentions = {
            'dontcare'
        }
        if self.value in none_mentions:
            gate = 'none'
        elif self.value in dontcare_mentions:
            gate = 'dontcare'
        else:
            gate = 'gen'
        return gate

    @property
    def is_empty(self) -> bool:
        return self.value == ''

    def __str__(self) -> str:
        return f"{self.domain} {self.slot} {self.value}".strip()
