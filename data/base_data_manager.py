from typing import NamedTuple, Optional

import pandas as pd
from carball.generated.api.game_pb2 import Game

from utils.utils import normalise_df


class GameData(NamedTuple):
    proto: Optional[Game]
    df: Optional[pd.DataFrame]


class DataManager:
    def __init__(self, need_proto: bool = False, need_df: bool = False, normalise_df: bool = True):
        self.need_proto = need_proto
        self.need_df = need_df
        self.normalise_df = normalise_df

    def get_data(self, id_: str) -> GameData:
        proto = self.get_proto(id_) if self.need_proto else None
        df = self.get_df(id_) if self.need_df else None
        if self.normalise_df:
            df = normalise_df(df)
        return GameData(proto, df)

    def get_replay_list(self, num: int = 50):
        raise NotImplementedError()

    def get_df(self, id_: str) -> pd.DataFrame:
        raise NotImplementedError()

    def get_proto(self, id_: str) -> Game:
        raise NotImplementedError()


class BrokenDataError(Exception):
    pass
