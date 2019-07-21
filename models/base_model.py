import logging
import time
from typing import Union

from tensorflow.python.keras.models import load_model, Sequential, Model, save_model

logger = logging.getLogger(__name__)


class BaseModel:
    def __init__(self, inputs: int, outputs: int, load_from_filepath: str = None, **kwargs):
        self.inputs = inputs
        self.outputs = outputs

        self.load_from_filepath = load_from_filepath  # Needed for copying with __dict__.
        if load_from_filepath is not None:
            self.model = load_model(load_from_filepath)
        else:
            self.model = self.build_model()

    def save_model(self, name: str = None, use_timestamp: bool = True):
        if name is None:
            name = self.__class__.__name__
        filename = f"model_{name}"
        if use_timestamp:
            filename += f"{time.strftime('%Y%m%d-%H%M%S')}"
        save_model(self.model, filename)

    def build_model(self) -> Union[Sequential, Model]:
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def fit_generator(self, *args, **kwargs):
        return self.model.fit_generator(*args, **kwargs)
