import datetime
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from carball.generated.api.game_pb2 import Game
from tensorflow.python.keras.callbacks import ModelCheckpoint

from data.interfacers.calculatedgg_api.api_interfacer import CalculatedApiInterfacer
from data.interfacers.calculatedgg_api.query_params import CalculatedApiQueryParams
from data.utils.columns import PlayerColumn, BallColumn, GameColumn
from data.utils.utils import filter_columns
from trainers.callbacks.metric_tracer import MetricTracer
from trainers.callbacks.tensorboard import get_tensorboard
from trainers.sequences.calculated_sequence import CalculatedSequence
from value_function.value_function_conv_model import ValueFunctionConvModel

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logging.getLogger("carball").setLevel(logging.CRITICAL)
logging.getLogger("data.base_data_manager").setLevel(logging.WARNING)


def get_input_and_output_from_game_datas(df: pd.DataFrame, proto: Game) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger.debug('Getting input and output')
    players = df.columns.levels[0]
    teams = proto.teams
    team_map = {
        player.id: team.is_orange
        for team in teams
        for player in team.player_ids
    }
    name_team_map = {player.name: player.is_orange for player in proto.players}

    sorted_players = sorted(
        [player for player in players if player not in ['ball', 'game']],
        key=lambda x: name_team_map[x]
    ) + ['ball', 'game']

    goal_teams_list = [team_map[goal.player_id.id] for goal in proto.game_metadata.goals]
    goal_frame_numbers = [goal.frame_number for goal in proto.game_metadata.goals]
    goal_times_list = df.loc[goal_frame_numbers, ('game', 'time')].tolist()

    # Get goal number
    goal_number = df.loc[:, ('game', 'goal_number')].dropna()
    goal_number = goal_number[goal_number >= 0]
    goal_team = goal_number.apply(lambda x: goal_teams_list[int(x)])
    goal_time = goal_number.apply(lambda x: goal_times_list[int(x)])

    # Only train on 1/3 of the frames - reduce extremely similar frames
    _df = df.sample(frac=0.3)

    _df[('game', 'goal_team')] = goal_team.rename('goal_team')
    _df[('game', 'goal_time')] = goal_time.rename('goal_time')
    _df[('game', 'time_to_goal')] = _df.loc[:, ('game', 'goal_time')] - _df.loc[:, ('game', 'time')]
    _df = _df.dropna(subset=[('game', 'goal_team')])

    # Remove post-goal frames
    _df = _df[_df.loc[:, ('game', 'time_to_goal')] >= 0]

    # _df = _df[sorted_players]  # Same thing as below line
    _df.reindex(columns=sorted_players, level=0)

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
    input_ = filter_columns(_df, INPUT_COLUMNS).fillna(0).astype(float)

    # Move value towards 0 if blue scored, or towards 1 if orange scored
    value_coefficient = ((-1) ** (_df.game.goal_team + 1)).astype(np.int8)

    MAX_TIME = 10

    # 0 (goal long later) to 1 (goal now)
    raw_value = ((MAX_TIME - _df.loc[:, ('game', 'time_to_goal')]) / MAX_TIME).clip(0, 1)
    output = 0.5 + 0.5 * raw_value * value_coefficient
    output = output.values.reshape((-1, 1))
    input_ = input_.values
    logger.debug(f'Got input and output: input shape: {input_.shape}, output shape:{output.shape}')
    return input_, output, get_sample_weight(output)


def get_sample_weight(output: np.ndarray) -> np.ndarray:
    weights = np.ones_like(output)
    # weights[output == 0.5] = 1 / np.sum(output == 0.5)
    weights[output == 0.5] = 1 / 5

    return weights.flatten()


INPUT_FEATURES = 61


if __name__ == '__main__':
    model = ValueFunctionConvModel(INPUT_FEATURES)
    # model = ValueFunctionModel(INPUT_FEATURES)

    interfacer = CalculatedApiInterfacer(
        CalculatedApiQueryParams(playlist=13, minmmr=1500, maxmmr=1550,
                                 start_timestamp=int(datetime.datetime(2018, 11, 1).timestamp()))
    )
    sequence = CalculatedSequence(
        interfacer=interfacer,
        game_data_transformer=get_input_and_output_from_game_datas,
    )

    EVAL_COUNT = 5
    eval_sequence = sequence.create_eval_sequence(EVAL_COUNT)
    eval_inputs, eval_outputs, _UNUSED_eval_weights = eval_sequence.as_arrays()

    save_callback = ModelCheckpoint('value_function.{epoch:02d}-{val_loss:.5f}.hdf5', save_best_only=True)
    callbacks = [
        MetricTracer(),
        save_callback,
        get_tensorboard(),
        # PredictionPlotter(model.model),
    ]
    model.fit_generator(sequence,
                        steps_per_epoch=500,
                        validation_data=eval_sequence, epochs=1000, callbacks=callbacks,
                        workers=4, use_multiprocessing=True)
