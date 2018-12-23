import logging
from typing import NamedTuple, Optional

import pandas as pd
from carball.generated.api.game_pb2 import Game

from data.utils.utils import normalise_df

logger = logging.getLogger(__name__)

class GameData(NamedTuple):
    proto: Optional[Game]
    df: Optional[pd.DataFrame]


class DataManager:
    """
    Abstract class that implements get_data() and the need_proto, need_df, and normalise_df attributes.
    Also implements the various methods required from subclasses.
    """
    def __init__(self, need_proto: bool = False, need_df: bool = False, normalise_df: bool = True):
        """
        :param need_proto: Whether to load the .proto attribute when get_data is called.
        :param need_df: Whether to load the .df attribute when get_data is called.
        :param normalise_df: Whether to normalise the df when get_data is called.
        """
        self.need_proto = need_proto
        self.need_df = need_df
        self.normalise_df = normalise_df

    def get_data(self, id_: str) -> GameData:
        """
        Returns a GameData object which has a .proto and .df attribute.
        Both default to None, unless self.need_proto or self.need_df are True respectively.
        If self.normalise_df is True, the returned GameData.df would be normalised.
        :param id_: Replay id
        :return: GameData object which has a .proto and .df attribute.
        """
        proto = self.get_proto(id_) if self.need_proto else None
        df = self.get_df(id_) if self.need_df else None
        if self.normalise_df:
            df = normalise_df(df)
        logger.info(f"Got data for replay: {id_}")
        return GameData(proto, df)

    def get_replay_list(self, num: int = 50):
        raise NotImplementedError()

    def get_df(self, id_: str) -> pd.DataFrame:
        raise NotImplementedError()

    def get_proto(self, id_: str) -> Game:
        raise NotImplementedError()


class BrokenDataError(Exception):
    pass
