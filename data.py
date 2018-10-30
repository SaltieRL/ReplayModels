import requests
import gzip
from carball.analysis.analysis_manager import PandasManager
from carball.analysis.utils.proto_manager import ProtobufManager

import io

BASE_URL = 'https://calculated.gg/api/v1/'


class Calculated:

    def get_replay_list(self, num=50):
        r = requests.get(BASE_URL + 'replays?key=1&minrank=19&teamsize=3')
        return r.json()['data']

    def get_pandas(self, id_):
        url = BASE_URL + 'parsed/{}.replay.gzip?key=1'.format(id_)
        r = requests.get(url)
        gzip_file = gzip.GzipFile(fileobj=io.BytesIO(r.content), mode='rb')

        pandas_ = PandasManager.safe_read_pandas_to_memory(gzip_file)
        return pandas_

    def get_proto(self, id_):
        url = BASE_URL + 'parsed/{}.replay.pts?key=1'.format(id_)
        r = requests.get(url)
        #     file_obj = io.BytesIO()
        #     for chunk in r.iter_content(chunk_size=1024):
        #         if chunk: # filter out keep-alive new chunks
        #             file_obj.write(chunk)
        proto = ProtobufManager.read_proto_out_from_file(io.BytesIO(r.content))
        return proto
