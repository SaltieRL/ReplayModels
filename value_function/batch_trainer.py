import logging
from typing import Tuple, Callable, List

import numpy as np
import pandas as pd
import torch
from carball.generated.api.game_pb2 import Game
from quicktracer import trace
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from data.base_data_manager import DataManager, BrokenDataError
from value_function.loss_plotter import LossPlotter
from .weighted_loss import WeightedMSELoss

logger = logging.getLogger(__name__)


class BatchTrainer:
    def __init__(self, data_manager: DataManager, model: nn.Module,
                 get_input_and_output_from_game_data: Callable[[pd.DataFrame, Game], Tuple[np.ndarray, np.ndarray]],
                 loss_criterion: _Loss, trace: bool = True, eval_set_length: int = 2, save_on_eval: bool = True):
        self.data_manager = data_manager
        self.model = model
        self.get_input_and_output_from_game_data = get_input_and_output_from_game_data
        self.trace: bool = trace
        self.eval_set_length: int = eval_set_length
        self.save_on_eval: bool = save_on_eval

        self.loss_criterion: _Loss = loss_criterion
        self.optimizer: Optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.eval_set: List[str] = []
        self.eval_inputs: np.ndarray = None
        self.eval_outputs: np.ndarray = None

        self.create_eval_set()

        self.replays_trained_on = 0
        self.batches_trained_on = 0

        self.loss_plotter = LossPlotter()

    def run(self, epochs: int = 200, replays_per_epoch: int = 200, replays_per_batch: int = 5, eval_every: int = 1):
        epoch_losses = []
        epoch_accuracies = []
        for epoch in range(epochs):
            input_train_batch = []
            output_train_batch = []
            replay_ids_batch = []

            epoch_loss = []
            epoch_accuracy = []
            for replay_id in self.data_manager.get_replay_list(num=replays_per_epoch, page=0):
                if replay_id in self.eval_set:
                    # Avoid training on replays in the test set
                    logger.info(f"Skipping replay as it's part of test set: {replay_id}")
                    continue
                try:
                    game_data = self.data_manager.get_data(replay_id)
                    df, proto = game_data.df, game_data.proto
                    if df is None or proto is None:
                        continue
                    input_train, output_train = self.get_input_and_output_from_game_data(df, proto)

                    input_train_batch.append(input_train)
                    output_train_batch.append(output_train)
                    replay_ids_batch.append(proto.game_metadata.match_guid)

                    if len(input_train_batch) >= replays_per_batch:
                        logger.info(f"Training on {len(replay_ids_batch)} replays.")
                        logger.debug(f":replay_ids_batch: {replay_ids_batch}")
                        try:
                            batch_loss = self._train(np.concatenate(input_train_batch),
                                                     np.concatenate(output_train_batch))
                            epoch_loss.append(batch_loss)
                            self.replays_trained_on += len(input_train_batch)
                            self.batches_trained_on += 1

                            if self.batches_trained_on % eval_every == 0:
                                self._evaluate()
                        except Exception as e:
                            logger.error(f"Unexpected error while training (or evaluating).")
                            import traceback
                            traceback.print_exc()
                            logger.error(e)
                        input_train_batch = []
                        output_train_batch = []
                        replay_ids_batch = []
                except BrokenDataError:
                    logger.warning(f"Replay: {replay_id} broken. Skipping.")
                except Exception as e:
                    logger.error(f"Unexpected error getting input data and output data for replay: {replay_id}.")
                    import traceback
                    traceback.print_exc()
                    logger.error(e)
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

    def _train(self, input_: np.ndarray, output: np.ndarray):
        input_tensor = torch.from_numpy(input_).float().cuda()
        output_tensor = torch.from_numpy(output).float().cuda()

        self.optimizer.zero_grad()
        logger.info(f"training on input of shape: {input_tensor.shape}")
        logger.debug(f'input_shape: {input_tensor.shape}')
        predicted = self.model(input_tensor)
        logger.debug(f'predicted_shape: {predicted.shape}')
        loss = self.loss_criterion(predicted, output_tensor)

        logger.debug(f"output shape: {output.shape}")

        self.loss_plotter.update_plot(output, predicted.data.cpu().numpy())

        loss.backward()
        self.optimizer.step()

        _loss = float(loss.cpu().data.numpy())
        if self.trace:
            trace(_loss)
        return _loss

    def create_eval_set(self):
        logger.info("Creating eval set.")
        eval_inputs = []
        eval_outputs = []
        for replay_id in self.data_manager.get_replay_list(num=self.eval_set_length, page=0):
            try:
                game_data = self.data_manager.get_data(replay_id)
                inputs_, outputs = self.get_input_and_output_from_game_data(game_data.df, game_data.proto)
                eval_inputs.append(inputs_)
                eval_outputs.append(outputs)
                self.eval_set.append(replay_id)
            except BrokenDataError:
                logger.warning(f"Error while adding replay {replay_id} to eval set")

        self.eval_inputs = np.concatenate(eval_inputs)
        self.eval_outputs = np.concatenate(eval_outputs)
        logger.info(f"Created eval set of {len(self.eval_set)} replays.")

    def _evaluate(self):
        # Set the model to evaluation mode (turns Dropout off)
        self.model.eval()

        with torch.no_grad():
            predicted = self.model(torch.from_numpy(self.eval_inputs).float().cuda())
            loss = self.loss_criterion(predicted, torch.from_numpy(self.eval_outputs).float().cuda())
            if self.trace:
                evaluation_loss = float(loss.cpu().data.numpy())
                trace(evaluation_loss)

            logger.info(f"evaluation loss: {loss}")
        # Return model to train mode
        self.model.train()
        if self.save_on_eval:
            torch.save(self.model, f"{self.model.__class__.__name__}.pt")
