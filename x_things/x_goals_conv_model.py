from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Reshape, Flatten
from tensorflow.python.keras.optimizers import Adam

from models.base_model import BaseModel


class XGoalsConvModel(BaseModel):
    def __init__(self, inputs: int, outputs: int, load_from_filepath: str = None):
        super().__init__(inputs, outputs, load_from_filepath)

    def build_model(self) -> Sequential:
        model = Sequential([
            Reshape((self.inputs, 1), input_shape=(self.inputs,)),
            Conv1D(filters=1024, kernel_size=9),
            Conv1D(filters=1024, kernel_size=6),
            Conv1D(filters=1024, kernel_size=3),
            Flatten(data_format='channels_last'),
            Dense(512, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.outputs, activation='sigmoid')
        ])

        optimizer = Adam(lr=1e-4)

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
