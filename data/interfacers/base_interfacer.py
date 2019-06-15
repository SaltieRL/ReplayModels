from typing import Set

import pandas as pd
from carball.generated.api.game_pb2 import Game


class BaseInterfacer:
    """
    Represents a set of replays for Sequences to interact with.
    """

    def get_total_count(self) -> int:
        raise NotImplementedError

    def get_all_replay_ids(self) -> Set[str]:
        raise NotImplementedError

    def get_replay_proto(self, replay_id: str) -> Game:
        raise NotImplementedError

    def get_replay_df(self, replay_id: str) -> pd.DataFrame:
        raise NotImplementedError

    def copy(self) -> 'BaseInterfacer':
        raise NotImplementedError

