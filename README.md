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

## data_main.py
Run this script to either download replay files, convert them to CSV, or combine CSVs into a dataset.
Doing this relies on the config.ini file in "data/"

Steps to getting a dataframe of replay data:

   Set up your config.ini file (what mode and mmr range you want to deal with, path options)
   
   In the command line with the necessary packages installed:
        
        python data_main.py      (to see what args you want to use)
        
        python data_main.py download [args]
        python data_main.py convert [args]
        python data_main.py dataset [args]

You now have a .h5 file that can be opened by pandas into a dataframe.



