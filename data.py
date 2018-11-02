import glob
import gzip
import io
import os
import pandas as pd

import carball
import requests
from carball.analysis.analysis_manager import PandasManager, AnalysisManager
from carball.analysis.utils.proto_manager import ProtobufManager
from carball.generated.api.game_pb2 import Game

BASE_URL = 'https://calculated.gg/api/v1/'


class DataManager:

    def get_replay_list(self, num: int = 50):
        raise NotImplementedError()

    def get_pandas(self, id_: str):
        raise NotImplementedError()

    def get_proto(self, id_: str):
        raise NotImplementedError()


class Calculated(DataManager):
    PANDAS_MAP = {}
    PROTO_MAP = {}
    BROKEN = []

    def get_replay_list(self, num=50, page=1):
        r = requests.get(BASE_URL + 'replays?key=1&minmmr=1300&maxmmr=1400&playlist=13&num={}&page={}'.format(num, page))
        return [replay['hash'] for replay in r.json()['data']]

    def get_pandas(self, id_):
        if id_ in self.BROKEN:
            return None
        if id_ in self.PANDAS_MAP:
            return self.PANDAS_MAP[id_]
        url = BASE_URL + 'parsed/{}.replay.gzip?key=1'.format(id_)
        r = requests.get(url)
        gzip_file = gzip.GzipFile(fileobj=io.BytesIO(r.content), mode='rb')
        try:
            pandas_ = PandasManager.safe_read_pandas_to_memory(gzip_file)
        except:
            self.PANDAS_MAP[id_] = None
            self.BROKEN.append(id_)
            return None
        self.PANDAS_MAP[id_] = pandas_
        return pandas_

    def get_proto(self, id_):
        if id_ in self.PROTO_MAP:
            return self.PROTO_MAP[id_]
        url = BASE_URL + 'parsed/{}.replay.pts?key=1'.format(id_)
        r = requests.get(url)
        #     file_obj = io.BytesIO()
        #     for chunk in r.iter_content(chunk_size=1024):
        #         if chunk: # filter out keep-alive new chunks
        #             file_obj.write(chunk)
        proto = ProtobufManager.read_proto_out_from_file(io.BytesIO(r.content))
        self.PROTO_MAP[id_] = proto
        return proto


class Carball(DataManager):
    REPLAYS_DIR = 'replays'
    REPLAYS_MAP = {}

    def get_replay_list(self, num=50):
        replays = glob.glob(os.path.join(self.REPLAYS_DIR, '*.replay'))
        return [os.path.basename(replay).split('.')[0] for replay in replays]

    def get_pandas(self, id_) -> pd.DataFrame:
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
