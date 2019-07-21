from typing import NamedTuple


class CalculatedApiQueryParams(NamedTuple):
    page: int = 1
    num: int = 200
    playlist: int = None
    minmmr: int = None
    maxmmr: int = None
    mmrany: bool = None
    minrank: int = None
    maxrank: int = None
    rankany: bool = None
    start_timestamp: float = None
    end_timestamp: float = None
    key: str = 'PLACEHOLDER'

    def copy(self):
        return self.__class__(**self._asdict())


