import logging
from collections import deque
from typing import Tuple, Sequence, Callable

import numpy as np
import pandas as pd
import torch
from carball.generated.api.game_pb2 import Game
from quicktracer import trace
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from base_data_manager import DataManager
from utils.utils import DataColumn
from weighted_loss import WeightedMSELoss

logger = logging.getLogger(__name__)


class BatchTrainer:
    def __init__(self, data_manager: DataManager, model: nn.Module,
                 get_input_and_output_from_game_data: Callable[[pd.DataFrame, Game], Tuple[np.ndarray, np.ndarray]],
                 trace: bool = True, test_ratio: float = 0.2):
        self.data_manager = data_manager
        self.model = model
        self.get_input_and_output_from_game_data = get_input_and_output_from_game_data
        self.trace = trace
        self.test_ratio = test_ratio

        self.loss_criterion: _Loss = WeightedMSELoss()
        self.optimizer: Optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.test_set = deque(maxlen=10)

    def run(self):
        EPOCHS = 200
        REPLAYS_PER_EPOCH = 200
        epoch_losses = []
        epoch_accuracies = []
        for epoch in range(EPOCHS):
            input_train_batch = []
            output_train_batch = []
            replay_ids_batch = []

            epoch_loss = []
            epoch_accuracy = []
            for replay_id in self.data_manager.get_replay_list(num=REPLAYS_PER_EPOCH, page=0):
                try:
                    game_data = self.data_manager.get_data(replay_id)
                    df, proto = game_data.df, game_data.proto
                    if df is None or proto is None:
                        continue
                    input_, output = self.get_input_and_output_from_game_data(df, proto)
                    input_train, input_test, output_train, output_test = self.split_into_test_train(input_, output)

                    input_train_batch.append(input_train)
                    output_train_batch.append(output_train)
                    replay_ids_batch.append(proto.game_metadata.match_guid)
                    self.test_set.append((input_test, output_test))

                    if len(input_train_batch) >= 5:
                        logger.info(f"Training on replays {replay_ids_batch}")

                        batch_loss = self._train(np.concatenate(input_train_batch), np.concatenate(output_train_batch))
                        epoch_loss.append(batch_loss)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    logger.error(e)
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

    def split_into_test_train(self, input_: np.ndarray, output: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rand = np.random.random(output.shape[0])
        mask = rand < self.test_ratio
        input_train, input_test = input_[mask], input_[~mask]
        output_train, output_test = output[mask], output[~mask]
        return input_train, input_test, output_train, output_test

    def _train(self, input_: np.ndarray, output: np.ndarray):
        input_tensor = torch.from_numpy(input_).float().cuda()
        output_tensor = torch.from_numpy(output).float().cuda()

        self.optimizer.zero_grad()
        logger.debug(f'input_shape: {input_tensor.shape}')
        predicted = self.model(input_tensor)
        logger.debug(f'predicted_shape: {predicted.shape}')
        loss = self.loss_criterion(predicted, output_tensor)

        print(loss)
        logger.debug(f"output shape: {output.shape}")
        loss.backward()
        self.optimizer.step()

        _loss = float(loss.cpu().data.numpy())
        if self.trace:
            trace(_loss)
        return _loss
