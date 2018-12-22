import gzip
import io

import pandas as pd
import requests
from carball.analysis.utils.pandas_manager import PandasManager
from carball.analysis.utils.proto_manager import ProtobufManager
from carball.generated.api.game_pb2 import Game

from data.base_data_manager import DataManager, BrokenDataError

BASE_URL = 'https://calculated.gg/api/v1/'


class CalculatedDM(DataManager):
    BROKEN = []

    def get_replay_list(self, num=50, page=1):
        r = requests.get(
            BASE_URL + 'replays?key=1&minmmr=1300&maxmmr=1400&playlist=13&num={}&page={}'.format(num, page))
        return [replay['hash'] for replay in r.json()['data']]

    def get_df(self, id_) -> pd.DataFrame:
        if id_ in self.BROKEN:
            raise BrokenDataError

        url = BASE_URL + 'parsed/{}.replay.gzip?key=1'.format(id_)
        r = requests.get(url)
        gzip_file = gzip.GzipFile(fileobj=io.BytesIO(r.content), mode='rb')
        try:
            pandas_ = PandasManager.safe_read_pandas_to_memory(gzip_file)
        except:
            self.BROKEN.append(id_)
            raise BrokenDataError
        return pandas_

    def get_proto(self, id_) -> Game:
        url = BASE_URL + 'parsed/{}.replay.pts?key=1'.format(id_)
        r = requests.get(url)

        proto = ProtobufManager.read_proto_out_from_file(io.BytesIO(r.content))
        return proto
