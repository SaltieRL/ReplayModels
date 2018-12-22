import glob
import os

import carball
import pandas as pd
from carball.analysis.analysis_manager import AnalysisManager
from carball.generated.api.game_pb2 import Game

from data.base_data_manager import DataManager


class CarballDM(DataManager):
    REPLAYS_DIR = 'replays'
    REPLAYS_MAP = {}

    def get_replay_list(self, num=50):
        replays = glob.glob(os.path.join(self.REPLAYS_DIR, '*.replay'))
        return [os.path.basename(replay).split('.')[0] for replay in replays]

    def get_df(self, id_) -> pd.DataFrame:
        return self._process(id_).data_frame

    def get_proto(self, id_) -> Game:
        return self._process(id_).protobuf_game

    def _process(self, id_) -> AnalysisManager:
        if id_ in self.REPLAYS_MAP:
            return self.REPLAYS_MAP[id_]
        path = os.path.join(self.REPLAYS_DIR, id_ + '.replay')
        manager = carball.analyze_replay_file(path, "replay.json")
        self.REPLAYS_MAP[id_] = manager
        return manager
