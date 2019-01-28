import logging
import random
from typing import List, Callable, Tuple, Union

import numpy as np
import pandas as pd
from carball.generated.api.game_pb2 import Game
from tensorflow.python.keras.utils import Sequence

from data.calculatedgg_api.api_interfacer import CalculatedApiInterfacer
from data.calculatedgg_api.errors import BrokenDataFrameError

logger = logging.getLogger(__name__)

GameDataToArraysTransformer = Callable[[pd.DataFrame, Game],
                                       Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]]


class CalculatedSequence(Sequence):
    def __init__(self, interfacer: CalculatedApiInterfacer,
                 game_data_transformer: GameDataToArraysTransformer,
                 replay_ids: List[str] = None):
        self.interfacer = interfacer
        self.game_data_transformer = game_data_transformer
        self.replay_ids: List[str] = replay_ids
        self._setup_sequence()

    def _setup_sequence(self):
        if self.replay_ids is None:
            total_count = self.interfacer.get_total_count()
            logger.info(f'Setting up sequence: total count: {total_count}')
            self.replay_ids = list(self.interfacer.get_all_replay_ids())
        logger.info(f'Created sequence with {len(self)} replay ids.')

    def __len__(self):
        return len(self.replay_ids)

    def __getitem__(self, index: int):
        replay_id = self.replay_ids[index]
        try:
            proto = self.interfacer.get_replay_proto(replay_id)
            df = self.interfacer.get_replay_df(replay_id)
            return self.game_data_transformer(df, proto)
        except BrokenDataFrameError:
            logger.warning(f'Replay {replay_id} has broken dataframe.')
        except Exception as e:
            logger.exception(e)

        # Try random replay.
        while True:
            replay_id = random.choice(self.replay_ids)
            try:
                proto = self.interfacer.get_replay_proto(replay_id)
                df = self.interfacer.get_replay_df(replay_id)
                return self.game_data_transformer(df, proto)
            except BrokenDataFrameError:
                logger.warning(f'Replay {replay_id} has broken dataframe.')
            except Exception as e:
                logger.exception(e)

    def create_eval_sequence(self, eval_count: int):
        """
        Creates a sequence to be used as evaluation.
        Removes the replays put into the evaluation sequence from the existing sequence.
        :param eval_count:
        :return:
        """
        eval_set = random.sample(self.replay_ids, eval_count)

        # Remove eval set from this set
        self.replay_ids = [replay_id for replay_id in self.replay_ids if replay_id not in eval_set]

        return self.__class__(interfacer=self.interfacer.copy(),
                              game_data_transformer=self.game_data_transformer,
                              replay_ids=eval_set)

    def as_arrays(self) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        logger.info(f"Generating arrays from sequence.")
        inputs = []
        outputs = []
        weights = []
        for i in range(len(self)):
            arrays = self[i]
            if len(arrays) == 2:
                input_, output = arrays
                inputs.append(input_)
                outputs.append(output)
            elif len(arrays) == 3:
                input_, output, sample_weights = arrays
                inputs.append(input_)
                outputs.append(output)
                weights.append(sample_weights)
            elif len(arrays) == 0:
                continue
            else:
                raise Exception(f"GameDataToArrayTransformer should return tuple of length 2 or 3, not {len(arrays)}.")

        logger.info(f"Generated arrays from {len(inputs)} arrays.")
        if len(weights):
            return np.concatenate(inputs), np.concatenate(outputs), np.concatenate(weights)
        else:
            return np.concatenate(inputs), np.concatenate(outputs)
