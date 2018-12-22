import gzip
import io
import json
import logging
import os
from typing import List

import pandas as pd
import requests
from carball.analysis.utils.pandas_manager import PandasManager
from carball.analysis.utils.proto_manager import ProtobufManager
from carball.generated.api.game_pb2 import Game

from data.base_data_manager import DataManager, BrokenDataError

BASE_URL = 'https://calculated.gg/api/v1/'
MIN_MMR = 1300


CACHE_FOLDER = "cache"

logger = logging.getLogger(__name__)


class CalculatedLocalDM(DataManager):
    def __init__(self, need_proto: bool = False, need_df: bool = False, cache_path: str = None):
        super().__init__(need_proto, need_df)

        # Ensure cache folder exists
        self.cache_path = cache_path if cache_path is not None else CACHE_FOLDER
        os.makedirs(self.cache_path, exist_ok=True)

        self.known_bad_ids_filepath = os.path.join(self.cache_path, "known_bad_ids.txt")
        if os.path.isfile(self.known_bad_ids_filepath):
            with open(self.known_bad_ids_filepath, 'r') as f:
                self.known_bad_ids = json.load(f)
                logger.info(f'Loaded {len(self.known_bad_ids)} known bad ids.')
                logger.debug(f'known_bad_ids: {self.known_bad_ids}')
        else:
            logger.info(f'Did not find any known bad ids')
            self.known_bad_ids = []

    def get_replay_list(self, num: int = 50, page: int = 1) -> List[str]:
        r = requests.get(BASE_URL + f'replays?key=1&minmmr={MIN_MMR}&maxmmr=3000&playlist=13&num={num}&page={page}')
        return [replay['hash'] for replay in r.json()['data']]

    def add_broken_id(self, id_: str):
        if id_ not in self.known_bad_ids:
            self.known_bad_ids.append(id_)
            with open(self.known_bad_ids_filepath, 'w') as f:
                json.dump(self.known_bad_ids, f)

    def get_df(self, id_: str) -> pd.DataFrame:
        if id_ in self.known_bad_ids:
            raise BrokenDataError

        cached_filepath = os.path.join(self.cache_path, id_ + '.replay.gzip')
        if os.path.exists(cached_filepath):
            gzip_file = gzip.GzipFile(cached_filepath, mode='rb')
            logger.info(f"Loaded {id_} df from cache.")
        else:
            url = BASE_URL + f'parsed/{id_}.replay.gzip?key=1'
            r = requests.get(url)
            file = io.BytesIO(r.content)
            gzip_file = gzip.GzipFile(fileobj=file, mode='rb')
            logger.info(f"Loaded {id_} df from site.")

        try:
            pandas_ = PandasManager.safe_read_pandas_to_memory(gzip_file)
            if pandas_ is None:
                raise BrokenDataError
            if not os.path.exists(cached_filepath):
                with open(cached_filepath, 'wb') as f:
                    f.write(r.content)
        except:
            self.add_broken_id(id_)
            raise BrokenDataError

        try:
            gzip_file.close()
            file.close()
        except NameError:
            pass
        return pandas_

    def get_proto(self, id_: str) -> Game:
        if id_ in self.known_bad_ids:
            raise BrokenDataError

        cached_filepath = os.path.join(self.cache_path, id_ + '.replay.pts')
        if os.path.exists(cached_filepath):
            file = open(cached_filepath, mode='rb')
            logger.info(f"Loaded {id_} proto from cache.")
        else:
            url = BASE_URL + f'parsed/{id_}.replay.pts?key=1'
            r = requests.get(url)
            file = io.BytesIO(r.content)
            logger.info(f"Loaded {id_} proto from site.")

            with open(cached_filepath, 'wb') as f:
                f.write(r.content)

        proto = ProtobufManager.read_proto_out_from_file(file)
        file.close()
        return proto
