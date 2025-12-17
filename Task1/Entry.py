from dataclasses import dataclass
from typing import Optional

@dataclass
class Entry:
    name: str
    birth_date: Optional[str]
    birth_place: Optional[str]