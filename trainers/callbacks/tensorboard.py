import logging
import os
import pathlib

from tensorflow.python.keras.callbacks import TensorBoard

LOG_FOLDER = "tensorboard_logs"
FOLDER_PREFIX = "run"

logger = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(dir_path, LOG_FOLDER)


def get_tensorboard() -> TensorBoard:
    try:
        i = find_max_run_number() + 1
    except (FileNotFoundError, ValueError):
        i = 0
    _log_path = os.path.join(log_path, f"{FOLDER_PREFIX}_{i}")
    pathlib.Path(_log_path).mkdir(parents=True, exist_ok=True)

    callback = TensorBoard(_log_path)
    logger.info(f"Created TensorBoard logs in {_log_path}.")
    return callback


def find_max_run_number() -> int:
    files = os.listdir(log_path)
    run_numbers = []
    for file in files:
        if file.startswith(FOLDER_PREFIX):
            run_number = int(file[len(FOLDER_PREFIX) + 1:])
            run_numbers.append(run_number)
    return max(run_numbers)
