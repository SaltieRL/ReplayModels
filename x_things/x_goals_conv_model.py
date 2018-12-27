from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Reshape, Flatten
from tensorflow.python.keras.optimizers import Adam

from models.base_model import BaseModel


class XGoalsConvModel(BaseModel):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(inputs, outputs)

    def build_model(self) -> Sequential:
        model = Sequential([
            Reshape((self.inputs, 1), input_shape=(self.inputs,)),
            Conv1D(filters=512, kernel_size=3),
            Flatten(data_format='channels_last'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.outputs, activation='sigmoid')
        ])

        optimizer = Adam(lr=1e-3)

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
