# ReplayModels

Creates models for Rocket League replay analysis.

## Implementation
### General classes and modules
Data retrieval is done through the `DataManager` subclasses (e.g. `CalculatedLocalDM`).
These subclasses expose a `get_data()` method which retrieves a `GameData` object (with `.df` and `.proto` attributes).

General utility functions such as filtering columns of the dataframe are available in `data/utils/utils.py`.

`data/utils/number_check.py`  checks the number of available replays in calculated.gg's api for a certain query,
for a given playlist and min MMR.

### `value_function`
Use `batched_value_function.py` which uses the refactorised class `BatchTrainer`.
Running it should cache replay dataframes and protos, and plot loss with quicktracer. 




