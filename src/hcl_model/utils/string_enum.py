from __future__ import annotations

from enum import Enum
from typing import List, ValuesView


class StringEnum(str, Enum):
    @classmethod
    def get_all_values(cls) -> List[str]:
        return [v.value for v in cls._get_all_values()]

    @classmethod
    def _get_all_members(cls) -> List:
        return list(cls._get_all_values())

    @classmethod
    def _get_all_values(cls) -> ValuesView:
        return cls.__members__.values()
