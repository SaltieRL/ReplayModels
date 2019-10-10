import glob
import os
import random
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime
import psutil
import pandas as pd
import numpy as np

# Getting config
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('data/config.ini')
mode = config['VARS']['MODE'].split(',')
mmrs = config['VARS']['mmr_range'].split('-')
NUM_PLAYERS = int(mode[1])
# Paths
paths = config['PATHS']
csv_path = paths['csv_path']
dataset_path = paths['dataset_path']
testcsv_path = paths['testcsv_path']
# CSV vars
cols_per_player = int(config['CSV']['columns_per_player'])
game_columns = int(config['CSV']['game_columns'])


def name_dataset(size, test):
    """
    Return a filename that reflects some context of the dataset creation.
    :param test: Whether the dataset is a test set.
    :type test: bool
    :param size: The number of games in the dataset.
    :type size: int
    :return: A filename.
    :rtype: str
    """
    if test:
        t = '-test_set'
    else:
        t = ''
    pcols = cols_per_player
    gcols = game_columns
    return f"{size}_games-{pcols}_pcols-{gcols}_gcols{t}"


# concat CSVs all together into a h5 dataset
# This overwrites the output file right now
def dataset(output_file=None, test=False, max_games=None, ram_max=75):
    """
    Concatenate CSVs designated by the config into a dataset.
    :param ram_max: When RAM usage hits this percentage, start a new chunk. Actual RAM usage at this point will only be half the max.
    :type ram_max: int (0-100)
    :param output_file: An output filename. Generated automatically if None.
    :type output_file: Optional[str]
    :param test: If the dataset will be a test set.
    :type test: bool
    :param max_games: Maximum number of games to put into the dataset.
    :type max_games: int
    :return: None
    :rtype: None
    """
    '''
    On chunking: The downside is that the shuffling is per chunk (frame level shuffling, not game shuffling),
        so with really small chunks the dataset won't be shuffled well.
    You'd think this would be fine since we can shuffle during training, but if you have RAM problems (4-16GB) that may not be the case.
    It's probably better to have a good shuffle (bigger chunks) now and then be able to train on smaller chunks later.
    '''
    # TODO: make ram_max smarter

    # Total size of team set
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print("Created directories in {}".format(dataset_path))
    if test and not os.path.exists(testcsv_path):
        print("Test is true but there is no testcsv_path")
        return
    if not os.path.exists(csv_path):
        print("There is no csv_path")
        return

    # Use testcsv_path for test data and csv_path for training data
    if test:
        input_glob = testcsv_path + '*.csv'
        if output_file is not None:
            output_file += '-test_set'
    else:
        input_glob = csv_path + '*.csv'
    dfs = glob.glob(input_glob)
    if (max_games is not None) and (max_games < len(dfs)):
        dfs = dfs[:max_games]
    if output_file is None:
        size = len(dfs)
        output_file = name_dataset(size, test)

    random.shuffle(dfs)
    new = True
    append_list = []
    bad_list = []
    chunk_time = datetime.now()
    # Easily get structure of columns with first df
    print("Starting")
    for df in dfs:
        game = pd.read_csv(df)
        game = game.drop('Unnamed: 0', axis=1)  # index_col arg is often problematic, this is an extra line but always works.
        append_list.append(game)
        # Repeat until done with chunk or out of df's.
        if psutil.virtual_memory().percent < ram_max and (df != dfs[-1]):
            continue
        else:
            print(psutil.virtual_memory())

        # Get here once every chunk
        result = pd.concat(append_list)
        append_list = []
        print("Sample")
        result = result.sample(frac=1)
        print("Write")
        if new:
            result.to_hdf(dataset_path + output_file + '.h5',
                          'data', mode='w', format='table', data_columns=['secs_to_goal', 'next_goal_one'])
            new = False
        else:
            result.to_hdf(dataset_path + output_file + '.h5',
                          'data', mode='a', format='table', append=True, data_columns=['secs_to_goal', 'next_goal_one'])
        print("--- chunk time: {} ---".format(datetime.now() - chunk_time))
        print("processed {} out of {} files".format(dfs.index(df) + 1, len(dfs)))
        chunk_time = datetime.now()
        del result

    return
