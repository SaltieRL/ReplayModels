from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizers import Adam

from models.base_model import BaseModel


class ValueFunctionModel(BaseModel):
    def __init__(self, inputs: int):
        super().__init__(inputs, 1)

    def build_model(self) -> Sequential:
        model = Sequential([
            Dense(512, input_dim=self.inputs, activation='relu'),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        optimizer = Adam(lr=1e-4)

        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        return model
