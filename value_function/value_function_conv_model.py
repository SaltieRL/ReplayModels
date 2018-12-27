from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Reshape, Conv1D, Flatten
from tensorflow.python.keras.optimizers import Adam

from models.base_model import BaseModel


class ValueFunctionConvModel(BaseModel):
    def __init__(self, inputs: int):
        super().__init__(inputs, 1)

    def build_model(self) -> Sequential:
        model = Sequential([
            Reshape((self.inputs, 1), input_shape=(self.inputs,)),
            Conv1D(filters=512, kernel_size=3),
            Flatten(data_format='channels_last'),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        optimizer = Adam(lr=1e-3)

        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        return model
