import logging
from typing import Tuple, Callable, List, Sequence, Set, Generator, Union

import numpy as np
import pandas as pd
from carball.generated.api.game_pb2 import Game
from tensorflow.python.keras.callbacks import Callback

from data.base_data_manager import DataManager, BrokenDataError
from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class BatchTrainer:
    def __init__(self, data_manager: DataManager, model: BaseModel,
                 get_input_and_output_from_game_data: Callable[[pd.DataFrame, Game], Tuple[np.ndarray, np.ndarray]],
                 get_sample_weight: Callable[[np.ndarray], np.ndarray] = None,
                 batch_size: int = 1024,
                 trace: bool = True, eval_set_length: int = 10, save_on_eval: bool = True):
        """

        :param data_manager:
        :param model:
        :param get_input_and_output_from_game_data:
            Function returning input and target output ndarrays, given the df and game.
        :param get_sample_weight:
            Function returning sample weights given the target output ndarray.
        :param batch_size:
        :param trace:
        :param eval_set_length:
        :param save_on_eval:
        """
        self.data_manager = data_manager
        self.model = model

        self.get_input_and_output_from_game_data = get_input_and_output_from_game_data
        self.get_sample_weight = get_sample_weight
        self.trace: bool = trace
        self.batch_size: int = batch_size
        self.eval_set_length: int = eval_set_length
        self.save_on_eval: bool = save_on_eval

        self.eval_set: List[str] = []
        self.eval_inputs: np.ndarray = None
        self.eval_outputs: np.ndarray = None

        self._create_eval_set()

        self.replays_trained_on: Set[str] = set()

    def run(self, epochs: int = 200, batches_per_epoch: int = 10, replays_per_batch: int = 5,
            minibatches_per_batch: int = 10, callbacks: Sequence[Callback] = None, verbose: int = 1):
        """
        Runs model.fit_generator with the relavant params.
        :param epochs: Number of epochs to run
        :param batches_per_epoch: Number of batches of replays that form an epoch
        :param replays_per_batch: Number of replays that constitute a batch
        :param minibatches_per_batch:
            Number of minibatches each batch is split into.
            Each batch is split into minibatches to perform multiple gradient updates per batch.
        :param callbacks:
        :param verbose:
        :return:
        """
        callbacks = [] if callbacks is None else callbacks
        for callback in callbacks:
            if callable(getattr(callback, "on_run_begin", None)):
                callback.on_run_begin(self.model.model)

        generator = self._get_generator(replays_per_batch, minibatches_per_batch)
        steps_per_epoch = minibatches_per_batch * batches_per_epoch
        self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                 validation_data=(self.eval_inputs, self.eval_outputs),
                                 shuffle=True, verbose=verbose, callbacks=callbacks, workers=5)

    def _create_eval_set(self):
        """
        Sets self.eval_set to list of ids, and sets self.eval_inputs and self.eval_outputs to np.ndarrays.
        :return: None
        """
        logger.info("Creating eval set.")
        eval_inputs = []
        eval_outputs = []
        while len(self.eval_set) < self.eval_set_length:
            for replay_id in self.data_manager.get_replay_list(num=self.eval_set_length):
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

    def _get_generator(self, replays_per_batch: int, minibatches_per_batch: int) \
            -> Generator[Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]], None, None]:
        """
        Returns generator to pass to fit_generator().
        :return:
            Generator yielding either (input, output) or (input, output, weight) as np.ndarrays,
            depending on if self.get_sample_weight is None.
        """

        def generator():
            while True:
                batch_ids = []
                batch_input = []
                batch_output = []
                for replay_id in self.data_manager.get_replay_list(num=200):
                    if replay_id in self.eval_set:
                        # Avoid training on replays in the test set
                        logger.info(f"Skipping replay as it's part of test set: {replay_id}")
                        continue
                    try:
                        game_data = self.data_manager.get_data(replay_id)
                        df, proto = game_data.df, game_data.proto
                        if df is None or proto is None:
                            continue
                        replay_input, replay_output = self.get_input_and_output_from_game_data(df, proto)

                        batch_input.append(replay_input)
                        batch_output.append(replay_output)
                        batch_ids.append(proto.game_metadata.match_guid)

                        if len(batch_ids) >= replays_per_batch:
                            logger.info(f"Generator yielding {len(batch_ids)} replays.")
                            logger.debug(f":replay_ids: {batch_ids}")

                            batch_input_array = np.concatenate(batch_input)
                            batch_output_array = np.concatenate(batch_output)
                            self.replays_trained_on.update(batch_ids)
                            logger.info(f"yielding input of shape: {batch_input_array.shape}, " +
                                        f"output of shape: {batch_output_array.shape}")

                            samples = len(batch_input_array)
                            # Shuffle arrays
                            permutation = np.random.permutation(samples)
                            batch_input_array = batch_input_array[permutation]
                            batch_output_array = batch_output_array[permutation]

                            # Split arrays
                            assert samples > minibatches_per_batch, \
                                f"Number of samples has to be greater than minibatches_per_batch " + \
                                f"({minibatches_per_batch}), found only {samples}"
                            minibatch_input_arrays = np.array_split(batch_input_array, minibatches_per_batch)
                            minibatch_output_arrays = np.array_split(batch_output_array, minibatches_per_batch)

                            for i in range(minibatches_per_batch):
                                minibatch_input = minibatch_input_arrays[i]
                                minibatch_output = minibatch_output_arrays[i]
                                logger.info(f"yielding minibatch input of shape: {minibatch_input.shape}, " +
                                            f"output of shape: {minibatch_output.shape}")
                                if self.get_sample_weight is not None:
                                    yield minibatch_input, minibatch_output, self.get_sample_weight(minibatch_output)
                                else:
                                    yield minibatch_input, minibatch_output

                            batch_input = []
                            batch_output = []
                            batch_ids = []
                    except BrokenDataError:
                        logger.warning(f"Replay: {replay_id} broken. Skipping.")
                    except Exception as e:
                        logger.error(f"Unexpected error getting input data and output data for replay: {replay_id}.")
                        import traceback
                        traceback.print_exc()
                        logger.error(e)

                        batch_input = []
                        batch_output = []
                        batch_ids = []

        return generator()
