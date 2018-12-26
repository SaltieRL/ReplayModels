from tensorflow.python.keras.callbacks import Callback
from quicktracer import trace


class MetricTracer(Callback):
    def on_batch_end(self, batch, logs=None):
        self._trace_logs(logs)

    def on_epoch_end(self, epoch, logs=None):
        self._trace_logs(logs)

    def _trace_logs(self, logs):
        for metric in self.params['metrics']:
            if metric in logs:
                trace(float(logs[metric]), key=metric)
