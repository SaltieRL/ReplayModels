from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizers import Adam

from models.base_model import BaseModel


class XThingsModel(BaseModel):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(inputs, outputs)

    def build_model(self) -> Sequential:
        model = Sequential([
            Dense(256, input_dim=self.inputs, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dense(self.outputs, activation='sigmoid')
        ])

        optimizer = Adam(lr=1e-3)

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
