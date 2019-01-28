import logging
from typing import Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_fscore_support
from tensorflow.python.keras.callbacks import Callback

logger = logging.getLogger(__name__)


class ClassificationMetrics(Callback):

    def __init__(self, validation_data_: Tuple[np.ndarray, np.ndarray]):
        super().__init__()
        self.validation_data_ = validation_data_

    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs=None):
        metrics = ['f1_metric', 'precision_metric', 'recall_metric']
        for metric in metrics:
            if metric not in self.params['metrics']:
                self.params['metrics'].append(metric)

        val_predict = (np.asarray(self.model.predict(self.validation_data_[0]))).round()
        val_targ = self.validation_data_[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_precision, _val_recall, _val_f1, support = precision_recall_fscore_support(val_targ, val_predict,
                                                                                        average='binary')
        logger.debug(f"Support (count of occurences in target): {support}")
        unique, counts = np.unique(val_predict, return_counts=True)
        logger.info(f"\nValidation predictions: unique: {unique}, counts: {counts}.")

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        # trace(_val_f1)
        # trace(_val_precision)
        # trace(_val_recall)
        if logs is not None:
            logs['f1_metric'] = _val_f1
            logs['precision_metric'] = _val_precision
            logs['recall_metric'] = _val_recall
        else:
            logger.warning('ClassificationMetrics not added to logs as logs is None.')
        return
