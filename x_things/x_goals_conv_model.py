from tensorflow.python.keras import Sequential, Input, Model
from tensorflow.python.keras.layers import Dense, Conv1D, Reshape, Flatten, Dot, Concatenate
from tensorflow.python.keras.optimizers import Adam

from models.base_model import BaseModel


class XGoalsConvModel(BaseModel):
    def __init__(self, inputs: int, outputs: int, load_from_filepath: str = None):
        super().__init__(inputs, outputs, load_from_filepath)

    def build_model(self) -> Sequential:
        model = Sequential([
            Reshape((self.inputs, 1), input_shape=(self.inputs,)),
            Conv1D(filters=1024, kernel_size=3, strides=3, padding="same"),
            Flatten(data_format='channels_last'),
            # Dense(1024, activation='relu', input_shape=(self.inputs,)),
            Dense(512, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.outputs, activation='sigmoid')
        ])

        optimizer = Adam(lr=1e-4)

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
