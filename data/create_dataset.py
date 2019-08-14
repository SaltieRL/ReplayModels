import glob
import os
import random
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime

import numpy as np
import pandas as pd

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


def sort_players(df, num_p, cols_per_p, game_cols):
    """
    Sort both teams in a game dataframe so that their players are in order of y position.
    :param df: The game dataframe.
    :type df: pd.DataFrame
    :param num_p: The number of players.
    :type num_p: int
    :param cols_per_p: The number of gdf columns per player.
    :type cols_per_p: int
    :param game_cols: The number of game columns.
    :type game_cols: int
    :return: A modified copy of the dataframe.
    :rtype: pd.DataFrame
    """
    team_cols = num_p * cols_per_p
    result = df.copy(deep=True)
    for i in range(len(result)):
        blue = pd.DataFrame(result.iloc[i][:team_cols].values.reshape(-1, cols_per_p)).sort_values(by=1).values.reshape(
            -1, team_cols)
        orange = pd.DataFrame(result.iloc[i][team_cols:(2 * team_cols)].values.reshape(-1, cols_per_p)).sort_values(
            by=1).values.reshape(-1, team_cols)
        game = pd.DataFrame(result.iloc[i][(2 * team_cols):]).values.reshape(-1, game_cols)
        array = np.concatenate((blue[0], orange[0], game[0]))
        result.iloc[i] = array
    return result


def name_dataset(test, size):
    """
    Return a filename that reflects some context of the dataset creation.
    :param test: Whether the dataset is a test set.
    :type test: bool
    :param size: The number of games in the dataset.
    :type size: int
    :return: A filename.
    :rtype: str
    """
    pcols = cols_per_player
    gcols = game_columns
    if test:
        t = "-test_set"
    else:
        t = ""
    return "{}_games-{}_pcols-{}_gcols{}.h5".format(size, pcols, gcols, t)


# concat CSVs all together into a h5 dataset
# This overwrites the output file right now
def dataset(num_players=NUM_PLAYERS, cols_per_p=cols_per_player, game_cols=game_columns, output_file=None, test=False,
            max_games=None, chunk_size=None):
    """
    Concatenate CSVs designated by the config into a dataset.
    :param num_players:
    :type num_players: int
    :param cols_per_p: The number of columns per player.
    :type cols_per_p: int
    :param game_cols: The number of game columns.
    :type game_cols: int
    :param output_file: An output filename. Generated automatically if None.
    :type output_file: Optional[str]
    :param test: If the dataset will be a test set.
    :type test: bool
    :param max_games: Maximum number of games to put into the dataset.
    :type max_games: int
    :param chunk_size: How much data we can manage in memory at a time. Use this if you are having RAM problems.
    :type chunk_size: int
    :return: None
    :rtype: None
    """
    '''
    On chunking: The downside is that the shuffling is per chunk (frame level shuffling, not game shuffling),
        so with really small chunks the dataset won't be shuffled well.
    You'd think this would be fine since we can shuffle during training, but if you have RAM problems (4-16GB) that may not be the case.
    It's probably better to have a good shuffle (bigger chunks) now and then be able to train on smaller chunks later.
    '''
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
    else:
        input_glob = csv_path + '*.csv'
    dfs = glob.glob(input_glob)
    if (max_games is not None) and (max_games < len(dfs)):
        dfs = dfs[:max_games]
    if output_file is None:
        size = len(dfs)
        output_file = name_dataset(test, size)
    elif test:
        output_file += "-test_set"
    random.shuffle(dfs)
    new = True
    chunk_count = 0
    append_list = []
    chunk_time = datetime.now()
    print("Starting")
    for df in dfs:
        game = pd.read_csv(df)
        game = game.drop('Unnamed: 0', axis=1)
        append_list.append(game)
        # Not mirroring right now
        # append_list.extend([game, mirror_df(game, num_players)])
        # Repeat until done with chunk or out of df's. "Or" statement is just for type checking of chunk_size basically.
        chunk_count += len(game)
        if ((chunk_size is None) or (chunk_count < chunk_size)) and (df != dfs[-1]):
            continue

        # Get here once every chunk
        result = pd.concat(append_list)
        append_list = []
        result = sort_players(result, num_players, cols_per_p, game_cols)
        result = result.sample(frac=1)

        if new:
            result.to_hdf(dataset_path + output_file + '.h5',
                          'data', mode='w', format='table')
            new = False
        else:
            result.to_hdf(dataset_path + output_file + '.h5',
                          'data', mode='a', format='table', append=True)
        print("--- chunk time: {} ---".format(datetime.now() - chunk_time))
        chunk_time = datetime.now()
        chunk_count = 0

    return
