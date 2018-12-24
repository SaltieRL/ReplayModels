import logging
from typing import Tuple

import numpy as np
import pandas as pd
from carball.generated.api.game_pb2 import Game
from carball.generated.api.player_pb2 import Player
from torch.nn import BCELoss

from data.calculated_local_dm import CalculatedLocalDM
from data.utils.columns import PlayerColumn, BallColumn, GameColumn
from data.utils.utils import filter_columns, flip_teams
from value_function.batch_trainer import BatchTrainer
from x_things.x_things_model import XThingsModel

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def get_input_and_output_from_game_datas(df: pd.DataFrame, proto: Game) -> Tuple[np.ndarray, np.ndarray]:
    logger.debug('Getting input and output')
    players = df.columns.levels[0]
    teams = proto.teams
    # team_map = {
    #     player.id: team.is_orange
    #     for team in teams
    #     for player in team.player_ids
    # }
    name_team_map = {player.name: player.is_orange for player in proto.players}

    sorted_players = sorted(
        [player for player in players if player not in ['ball', 'game']],
        key=lambda x: name_team_map[x]
    ) + ['ball', 'game']

    _df = df.copy()

    # _df = _df[sorted_players]  # Same thing as below line
    _df.reindex(columns=sorted_players[::-1], level=0)

    # Set up data
    INPUT_COLUMNS = [
        PlayerColumn.POS_X, PlayerColumn.POS_Y, PlayerColumn.POS_Z,
        PlayerColumn.ROT_X, PlayerColumn.ROT_Y, PlayerColumn.ROT_Z,
        PlayerColumn.VEL_X, PlayerColumn.VEL_Y, PlayerColumn.VEL_Z,
        # PlayerColumn.ANG_VEL_X, PlayerColumn.ANG_VEL_Y, PlayerColumn.ANG_VEL_Z,
        BallColumn.POS_X, BallColumn.POS_Y, BallColumn.POS_Z,
        BallColumn.VEL_X, BallColumn.VEL_Y, BallColumn.VEL_Z,
        GameColumn.SECONDS_REMAINING
    ]
    filtered_df = filter_columns(_df, INPUT_COLUMNS).fillna(0).astype(float)
    filtered_df_orange = flip_teams(filtered_df)

    name_map = {
        player.id.id: player.name
        for player in proto.players
    }

    hits = proto.game_stats.hits
    inputs = []
    outputs = []
    for hit in hits:
        player_name = name_map[hit.player_id.id]

        player = [player for player in proto.players if player.name == player_name][0]

        # Make player taking shot be blue
        _df = filtered_df_orange if player.is_orange else filtered_df
        # Get right frame
        frame = _df.loc[hit.frame_number, :]

        # Move player taking shot
        def key_fn(player_name: str) -> int:
            # Move player to front, move team to front.
            if player_name == player.name:
                return 0
            elif name_team_map[player_name] == player.is_orange:
                return 1
            else:
                return 2

        sorted_players = sorted(
            [player for player in players if player not in ['ball', 'game']],
            key=key_fn
        ) + ['ball', 'game']
        frame.reindex(sorted_players[::-1], level=0)

        inputs.append(frame.values)
        hit_output = [bool(getattr(hit, category)) for category in HIT_CATEGORIES]
        outputs.append(hit_output)

    input_ = np.array(inputs, dtype=np.float32)
    output = np.array(outputs, dtype=np.float32)

    logger.debug(f'Got input and output: input shape: {input_.shape}, output shape:{output.shape}')
    assert not np.any(np.isnan(input_)), "input contains nan"
    assert not np.any(np.isnan(output)), "output contains nan"
    return input_, output


data_manager = CalculatedLocalDM(need_df=True, need_proto=True,
                                 cache_path=r"C:\Users\harry\Documents\rocket_league\ReplayModels\cache")
INPUT_FEATURES = 61
# HIT_CATEGORIES = ['pass_', 'passed', 'dribble', 'dribble_continuation', 'shot', 'goal', 'assist', 'assisted',
#                   'save', 'aerial']
# HIT_CATEGORIES = ['pass_', 'shot', 'goal', 'aerial']
HIT_CATEGORIES = ['goal']
model = XThingsModel(INPUT_FEATURES, len(HIT_CATEGORIES)).cuda().train()
trainer = BatchTrainer(data_manager, model, get_input_and_output_from_game_datas, BCELoss())

trainer.run()
