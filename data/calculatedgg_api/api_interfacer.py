import gzip
import io
import logging
import pandas as pd
from typing import List, Set

import requests
from carball.analysis.utils.pandas_manager import PandasManager
from carball.analysis.utils.proto_manager import ProtobufManager
from carball.generated.api.game_pb2 import Game
from requests import Response

from data.calculatedgg_api.errors import BrokenDataFrameError
from data.calculatedgg_api.query_params import CalculatedApiQueryParams

logger = logging.getLogger(__name__)


class CalculatedApiInterfacer:
    BASE_URL = 'https://calculated.gg/api/v1/'

    def __init__(self, query_params: CalculatedApiQueryParams = CalculatedApiQueryParams()):
        self.initial_query_params = query_params
        self.query_params = query_params.copy()

    def get_total_count(self) -> int:
        r = self._get_replays_request(self.initial_query_params)
        r_json = r.json()
        total_count = r_json['total_count']
        logger.debug(f'Found a total of {total_count} replays.')
        return total_count

    def get_all_replay_ids(self) -> Set[str]:
        replay_ids = set()
        _query_params = self.initial_query_params._replace(page=1)
        while True:
            replay_ids_on_page = self.get_replay_list(_query_params)
            if len(replay_ids_on_page) == 0:
                break
            _query_params = _query_params._replace(page=_query_params.page + 1)
            replay_ids.update(replay_ids_on_page)
        logger.info(f'Found a total of {len(replay_ids)} unique replay ids.')
        return replay_ids

    def get_replay_list(self, query_params: CalculatedApiQueryParams = None) -> List[str]:
        query_params = query_params if query_params is not None else self.query_params
        r = self._get_replays_request(query_params)
        return [replay['hash'] for replay in r.json()['data']]

    @classmethod
    def _get_replays_request(cls, query_params: CalculatedApiQueryParams) -> Response:
        r = requests.get(cls.BASE_URL + 'replays', params=query_params._asdict())
        r.raise_for_status()
        logger.debug(f'Performed request for {r.url}')
        return r

    def get_replay_proto(self, replay_id: str) -> Game:
        url = self.BASE_URL + f'parsed/{replay_id}.replay.pts'
        r = requests.get(url, params={'key': self.query_params.key})
        r.raise_for_status()
        file = io.BytesIO(r.content)
        proto = ProtobufManager.read_proto_out_from_file(file)
        logger.debug(f"Loaded {replay_id} proto from site.")
        return proto

    def get_replay_df(self, replay_id: str) -> pd.DataFrame:
        url = self.BASE_URL + f'parsed/{replay_id}.replay.gzip'
        r = requests.get(url, params={'key': self.query_params.key})
        r.raise_for_status()
        gzip_file = gzip.GzipFile(fileobj=io.BytesIO(r.content), mode='rb')
        df = PandasManager.safe_read_pandas_to_memory(gzip_file)
        if df is None:
            raise BrokenDataFrameError
        logger.debug(f"Loaded {replay_id} df from site.")
        return df

    def copy(self):
        return self.__class__(self.query_params.copy())
