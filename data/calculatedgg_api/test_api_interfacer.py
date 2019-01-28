import datetime
import pandas as pd
import unittest

from carball.generated.api.game_pb2 import Game
from requests import HTTPError

from data.calculatedgg_api.api_interfacer import CalculatedApiInterfacer
from data.calculatedgg_api.query_params import CalculatedApiQueryParams


class ApiInterfacerTest(unittest.TestCase):

    def setUp(self) -> None:
        self.interfacer = CalculatedApiInterfacer()

    def test_default_get_replay_list(self):
        replay_ids = self.interfacer.get_replay_list()
        self.assertGreater(len(replay_ids), 0)

    def test_get_total_count(self):
        total_count = self.interfacer.get_total_count()
        self.assertGreater(total_count, 0)

    def test_get_all_replay_ids(self):
        query_params = CalculatedApiQueryParams(
            playlist=13,
            minmmr=1800,
            start_timestamp=int(datetime.datetime(2019, 1, 1).timestamp()),
            end_timestamp=int(datetime.datetime(2019, 1, 5).timestamp()),
        )
        interfacer = CalculatedApiInterfacer(query_params)
        total_count = interfacer.get_total_count()
        self.assertGreater(total_count, 0)
        all_replays = interfacer.get_all_replay_ids()
        self.assertEqual(total_count, len(all_replays))

    def test_get_replay_data_from_id(self):
        replay_id = "96BDDDEE11E8B6D4396D1B9668244BC6"  # actually exists
        proto: Game = self.interfacer.get_replay_proto(replay_id)
        df = self.interfacer.get_replay_df(replay_id)

        self.assertEqual(replay_id, proto.game_metadata.match_guid)
        self.assertIsInstance(df, pd.DataFrame)

        replay_id = "MADE_UP_THING"  # Does not exist
        with self.assertRaises(HTTPError):
            proto: Game = self.interfacer.get_replay_proto(replay_id)
        with self.assertRaises(HTTPError):
            df = self.interfacer.get_replay_df(replay_id)


if __name__ == '__main__':
    unittest.main()
