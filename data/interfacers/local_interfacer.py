import glob
import logging
import os
from typing import Set, Dict

import carball
import pandas as pd
from carball.analysis.utils.pandas_manager import PandasManager
from carball.analysis.utils.proto_manager import ProtobufManager
from carball.generated.api.game_pb2 import Game

from data.interfacers.base_interfacer import BaseInterfacer

logger = logging.getLogger(__name__)


class LocalInterfacer(BaseInterfacer):
    CACHE_PATH = r"C:\Users\harry\Documents\rocket_league\ReplayModels\cache"

    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.replay_paths: Dict[str, str] = self._get_replay_paths()  # replay id to replay path

        self.known_bad_ids = []

    def _get_replay_paths(self) -> Dict[str, str]:
        replays = glob.glob(os.path.join(self.folder_path, '**/*.replay'), recursive=True)
        replay_paths = {
            os.path.basename(replay_path)[:-7]: replay_path  # -7 to remove ".replay"
            for replay_path in replays
        }
        return replay_paths

    def get_total_count(self) -> int:
        return len(self.replay_paths)

    def get_all_replay_ids(self) -> Set[str]:
        replay_ids = set(self.replay_paths.keys())
        logger.info(f'Found a total of {len(replay_ids)} unique replay ids.')
        return replay_ids

    def get_replay_proto(self, replay_id: str) -> Game:
        proto_filename = self._get_proto_filename(replay_id)
        if proto_filename in os.listdir(self.CACHE_PATH):
            with open(os.path.join(self.CACHE_PATH, proto_filename), 'rb') as f:
                proto = ProtobufManager.read_proto_out_from_file(f)
            return proto
        else:
            proto = self.parse_replay(replay_id, return_proto=True)
            return proto

    def get_replay_df(self, replay_id: str) -> pd.DataFrame:
        dataframe_filename = self._get_dataframe_filename(replay_id)

        if dataframe_filename in os.listdir(self.CACHE_PATH):
            with open(os.path.join(self.CACHE_PATH, dataframe_filename), 'rb') as f:
                dataframe = PandasManager.safe_read_pandas_to_memory(f)
                if dataframe is None:
                    self.known_bad_ids.append(replay_id)
                    raise Exception(f'Cannot read replay dataframe {replay_id}')
            return dataframe
        else:
            dataframe = self.parse_replay(replay_id, return_dataframe=True)
            return dataframe

    def parse_replay(self, replay_id: str, return_proto: bool = False, return_dataframe: bool = False):
        assert not (return_proto and return_dataframe), 'Cannot return both proto and dataframe'
        replay_path = self.replay_paths[replay_id]
        try:
            analysis_manager = carball.analyze_replay_file(replay_path)
            proto_filename = self._get_proto_filename(replay_id)
            proto_filepath = os.path.join(self.CACHE_PATH, proto_filename)
            with open(proto_filepath, 'wb') as f:
                analysis_manager.write_proto_out_to_file(f)

            dataframe_filename = self._get_dataframe_filename(replay_id)
            dataframe_filepath = os.path.join(self.CACHE_PATH, dataframe_filename)
            with open(dataframe_filepath, 'wb') as f:
                analysis_manager.write_pandas_out_to_file(f)

            if return_proto:
                return analysis_manager.protobuf_game
            if return_dataframe:
                return analysis_manager.data_frame
        except Exception as e:
            print(f'Failed to parse replay: {e}')
            self.known_bad_ids.append(replay_id)

    @staticmethod
    def _get_proto_filename(replay_id: str):
        return replay_id + '.proto'

    @staticmethod
    def _get_dataframe_filename(replay_id: str):
        return replay_id + '.df'

    def copy(self) -> 'LocalInterfacer':
        return self.__class__(folder_path=self.folder_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    folder = r"C:\Users\harry\Documents\rocket_league\replays\DHPC DHE19 Replay Files"
    folder = r"C:\Users\harry\Documents\rocket_league\replays\RLCS Season 6"
    interfacer = LocalInterfacer(folder)
    replay_ids = interfacer.get_all_replay_ids()

    print(replay_ids)
    replay_id = sorted(list(replay_ids))[0]
    proto: Game = interfacer.get_replay_proto(replay_id)

    print(len(proto.players))
