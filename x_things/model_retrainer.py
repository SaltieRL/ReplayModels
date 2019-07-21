from functools import partial

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Adam

from data.interfacers.local_interfacer import LocalInterfacer
from trainers.callbacks.metric_tracer import MetricTracer
from trainers.callbacks.metrics import ClassificationMetrics
from trainers.callbacks.tensorboard import get_tensorboard
from trainers.sequences.calculated_sequence import CalculatedSequence
from x_things.x_goals_trainer import get_input_and_output_from_game_datas, WeightMethod



if __name__ == '__main__':

    filepath = "x_goals.113-0.88718.hdf5"


    model = load_model(filepath)

    # Set only last layer to be trainable
    for layer in model.layers[:-2]:
        layer.trainable = False

    for layer in model.layers:
        print(layer, layer.trainable)
    print(model.summary())

    optimizer = Adam(lr=1e-5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # K.set_value(model.optimizer.lr, 1e-5)

    folder = r"C:\Users\harry\Documents\rocket_league\replays\DHPC DHE19 Replay Files"
    interfacer = LocalInterfacer(folder)
    retraining_sequence = CalculatedSequence(
        interfacer=interfacer,
        game_data_transformer=partial(get_input_and_output_from_game_datas, weight_method=WeightMethod.EMPHASISE_SHOTS),
    )

    EVAL_COUNT = 50
    eval_sequence = retraining_sequence.create_eval_sequence(EVAL_COUNT)
    eval_inputs, eval_outputs, _ = eval_sequence.as_arrays()

    save_callback = ModelCheckpoint('x_goals_retrained.{epoch:02d}-{val_acc:.5f}.hdf5', monitor='val_acc', save_best_only=True)
    classificaion_metrics = ClassificationMetrics((eval_inputs, eval_outputs))
    callbacks = [
        classificaion_metrics,
        MetricTracer(),
        save_callback,
        get_tensorboard(),
        # PredictionPlotter(model.model),
    ]
    model.fit_generator(retraining_sequence,
                        steps_per_epoch=500,
                        validation_data=eval_sequence, epochs=1000, callbacks=callbacks,
                        workers=4, use_multiprocessing=True)
