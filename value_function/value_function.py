import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from carball.generated.api.game_pb2 import Game
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from quicktracer import trace

from base_data_manager import GameData
from calculated_local_dm import CalculatedLocalDM
from data.calulated_dm import CalculatedDM
from goal_predictor import GoalPredictor
from utils.columns import PlayerColumn, BallColumn, GameColumn
from utils.utils import filter_columns
from weighted_loss import WeightedMSELoss

logger = logging.getLogger(__name__)


def get_input_output_from_game_data(df: pd.DataFrame, proto: Game) -> Tuple[np.ndarray, np.ndarray]:
    logger.info('Getting input and output')
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

    _df = df.copy()
    _df[('game', 'goal_team')] = goal_team.rename('goal_team')
    _df[('game', 'goal_time')] = goal_time.rename('goal_time')
    _df[('game', 'time_to_goal')] = _df.loc[:, ('game', 'goal_time')] - _df.loc[:, ('game', 'time')]
    _df = _df.dropna(subset=[('game', 'goal_team')])

    # Remove post-goal frames
    _df = _df[_df['game']['time_to_goal'] >= 0]

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
    input_ = filter_columns(_df, INPUT_COLUMNS).fillna(0).astype(float)

    # Move value towards 0 if blue scored, or towards 1 if orange scored
    value_coefficient = ((-1) ** (_df.game.goal_team + 1)).astype(np.int8)

    MAX_TIME = 10

    # 0 (goal long later) to 1 (goal now)
    raw_value = ((MAX_TIME - _df.loc[:, ('game', 'time_to_goal')]) / MAX_TIME).clip(0, 1)
    output = 0.5 + 0.5 * raw_value * value_coefficient
    output = output.values.reshape((-1, 1))
    input_ = input_.values
    logger.info(f'Got input and output: input shape: {input_.shape}, output shape:{output.shape}')
    return input_, output


def split_into_test_train(input_: np.ndarray, output: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rand = np.random.random(output.shape[0])
    mask = rand < 0.8
    input_train, input_test = input_[mask], input_[~mask]
    output_train, output_test = output[mask], output[~mask]
    return input_train, input_test, output_train, output_test


def train_on_proto_df(proto: Game, df: pd.DataFrame, model: nn.Module,
                      loss_criterion: _Loss, optimizer: Optimizer):
    logger.info(f"Training on proto {proto.game_metadata.match_guid}")
    input_, output = get_input_output_from_game_data(df, proto)
    input_train, input_test, output_train, output_test = split_into_test_train(input_, output)

    input_tensor = torch.from_numpy(input_train).float().cuda()
    output_tensor = torch.from_numpy(output_train).float().cuda()

    optimizer.zero_grad()
    logger.debug(f'input_shape: {input_tensor.shape}')
    predicted = model(input_tensor)
    logger.debug(f'predicted_shape: {predicted.shape}')
    loss = loss_criterion(predicted, output_tensor)

    print(loss)
    logger.debug(f"output_train shape: {output_train.shape}")
    loss.backward()
    optimizer.step()

    _loss = float(loss.cpu().data.numpy())
    trace(_loss)
    return _loss


def main():
    logging.basicConfig(level=logging.DEBUG)
    # data_manager = CalculatedDM(need_df=True, need_proto=True)
    data_manager = CalculatedLocalDM(need_df=True, need_proto=True)

    INPUT_FEATURES = 61

    columns_to_keep = [
        PlayerColumn.POS_X, PlayerColumn.POS_Y, PlayerColumn.POS_Z,
        PlayerColumn.ROT_X, PlayerColumn.ROT_Y, PlayerColumn.ROT_Z,
        PlayerColumn.VEL_X, PlayerColumn.VEL_Y, PlayerColumn.VEL_Z,
        # PlayerColumn.ANG_VEL_X, PlayerColumn.ANG_VEL_Y, PlayerColumn.ANG_VEL_Z,
        BallColumn.POS_X, BallColumn.POS_Y, BallColumn.POS_Z,
        BallColumn.VEL_X, BallColumn.VEL_Y, BallColumn.VEL_Z,
        GameColumn.SECONDS_REMAINING, GameColumn.GOAL_NUMBER,
        GameColumn.TIME
    ]

    required_columns = len([column for column in columns_to_keep if isinstance(column, PlayerColumn)]) * 6 + \
                       len([column for column in columns_to_keep if not isinstance(column, PlayerColumn)])

    model = GoalPredictor(INPUT_FEATURES).cuda().train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # loss = nn.MSELoss()
    loss = WeightedMSELoss()

    EPOCHS = 200
    REPLAYS_PER_EPOCH = 200
    epoch_losses = []
    epoch_accuracies = []
    for epoch in range(EPOCHS):
        epoch_loss = []
        epoch_accuracy = []
        for replay_id in data_manager.get_replay_list(num=REPLAYS_PER_EPOCH, page=0):
            try:
                game_data = data_manager.get_data(replay_id)
                df, proto = game_data.df, game_data.proto
                if df is None or proto is None:
                    continue
                df = filter_columns(df, columns_to_keep)

                if len(df.columns) != required_columns:
                    logger.warning(f"Replay df does not have correct number of columns: {replay_id}. " +
                                   f"Should be {required_columns}, found {len(df.columns)}.")
                    continue
                proto_loss = train_on_proto_df(proto, df, model, loss, optimizer)
                epoch_loss.append(proto_loss)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(e)
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

    means = []
    for l in epoch_losses:
        l = np.array(l)
        l = l[l != None]
        means.append(l.mean())

    plt.plot(range(len(epoch_losses)), means)
    plt.title('Loss')

    means = []
    for l in epoch_accuracies:
        l = np.array(l)
        #     print(l)
        #     l = l[l != None]
        means.append([l[:, 0].mean(), l[:, 1].mean()])
    plt.plot(range(len(epoch_accuracies)), means)
    plt.title('Accuracy')
    plt.legend(['Team Accuracy', 'Frame Accuracy'])

    input_train, input_test, output_train, output_test = get_train_test_data(proto, df)

    test_output = model(torch.from_numpy(input_test).float().cuda())
    output = test_output.cpu().detach().numpy()
    team_accuracy = (output[:, 0].round() == output_test[:, 0]).sum() / output.shape[0]
    frame_accuracy = (abs(output[:, 1] - output_test[:, 1]) < 30 * 1).sum() / output.shape[0]

    plt.scatter(output_test[output_test[:, 1] > 0][:, 1], output[output_test[:, 1] > 0][:, 1]);
    plt.xlabel('time to goal')
    plt.ylabel('predicted')


if __name__ == '__main__':
    main()
