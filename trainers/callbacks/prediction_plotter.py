import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import Callback

from trainers.callbacks.prediction_plotter_plot import PredictionPlotterPlot

logger = logging.getLogger(__name__)


class PredictionPlotter(Callback):
    """
    See https://stackoverflow.com/a/47081613 for implementation idea.

    An "on_run_begin(model)" method is defined as the initialisation has to be called before fit()
    (and thus before any of the existing methods of the base class, including even set_model())
    """

    def __init__(self, plot_every_x_steps: int = 20):
        super().__init__()
        self.plot_every_x_steps = plot_every_x_steps
        self.plot = PredictionPlotterPlot()

        self.targets = []  # collect y_true batches
        self.outputs = []  # collect y_pred batches

        # the shape of these 2 variables will change according to batch shape
        # to handle the "last batch", specify `validate_shape=False`
        self.var_y_true = tf.Variable(0., validate_shape=False)
        self.var_y_pred = tf.Variable(0., validate_shape=False)

    def _initialise_variables(self, model):
        fetches = [tf.assign(self.var_y_true, model.targets[0], validate_shape=False),
                   tf.assign(self.var_y_pred, model.outputs[0], validate_shape=False)]

        model._function_kwargs = {'fetches': fetches}
        # use `model._function_kwargs` if using `Model` instead of `Sequential`

    def on_run_begin(self, model: Sequential):
        self._initialise_variables(model)

    def on_batch_end(self, batch, logs=None):
        self.targets.append(K.eval(self.var_y_true))
        self.outputs.append(K.eval(self.var_y_pred))

        if batch % self.plot_every_x_steps == 0:
            self.update_plot()

    def update_plot(self):
        actual = np.concatenate(self.targets, axis=None)
        predicted = np.concatenate(self.outputs, axis=None)
        logger.info(f"Plotting prediction on {len(predicted)} samples in epoch")
        self.plot.update_plot(actual, predicted)
        self.targets = []
        self.outputs = []
