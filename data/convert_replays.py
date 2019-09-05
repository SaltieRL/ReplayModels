import multiprocessing as mp
import os
import random
import sys
import time
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime, timedelta
from multiprocessing import Process, Value, current_process
from shutil import copyfile
from typing import List

import carball
import numpy as np
import pandas as pd
from carball.rattletrap.run_rattletrap import RattleTrapException

# Getting config
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('data/config.ini')
mode = config['VARS']['MODE'].split(',')
mmrs = config['VARS']['mmr_range'].split('-')
NUM_PLAYERS = int(mode[1])
# Paths
paths = config['PATHS']
# Paths for files we are skipping for various reasons
error_path = paths['error_path']
skip_path = paths['skip_path']
# PATHVARS specific to the number of players
replay_path = paths['replay_path']
json_path = paths['json_path']
csv_path = paths['csv_path']
dataset_path = paths['dataset_path']
testcsv_path = paths['testcsv_path']


# HELPERS
# STOPPING RATTLETRAPS PRINTS (https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python)
class HiddenPrints:
    """
    Context manager that prevents printing to stdout or stderr.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


""" with HiddenPrints():
        print("This will not be printed")
    ...
    print("This will be printed as before")"""


# Helper: Ordering the columns of the df
def get_ordered_columns(players_per_team: int) -> List[str]:
    """
    Return an ordered list of column names to be passed to a game dataframe.
    :param players_per_team: Determines how many player columns to return.
    :type players_per_team: int
    :return: A list of column names.
    :rtype: List[str]
    """
    x = players_per_team
    non_player = ['ball_pos_x', 'ball_pos_y', 'ball_pos_z', 'ball_rot_x', 'ball_rot_y', 'ball_rot_z',
                  'ball_vel_x', 'ball_vel_y', 'ball_vel_z', 'ball_ang_vel_x', 'ball_ang_vel_y', 'ball_ang_vel_z',
                  'game_seconds_remaining']
    z_zero = ['z_0_pos_x', 'z_0_pos_y', 'z_0_pos_z', 'z_0_rot_x', 'z_0_rot_y', 'z_0_rot_z',
              'z_0_vel_x', 'z_0_vel_y', 'z_0_vel_z', 'z_0_ang_vel_x', 'z_0_ang_vel_y', 'z_0_ang_vel_z',
              'z_0_boost', 'z_0_boost_active', 'z_0_jump_active', 'z_0_double_jump_active', 'z_0_dodge_active',
              'z_0_is_demo']
    z_one = []
    z_two = []
    o_zero = []
    o_one = []
    o_two = []
    for col in z_zero:
        if x > 1:
            z_one.append(col.replace('0', '1', 1))
        if x > 2:
            z_two.append(col.replace('0', '2', 1))
        o_zero.append(col.replace('z', 'o', 1))
    if x > 1:
        for col in o_zero:
            o_one.append(col.replace('0', '1', 1))
            if x > 2:
                o_two.append(col.replace('0', '2', 1))

    columns_ordered = z_zero + z_one + z_two + o_zero + o_one + o_two + non_player
    return columns_ordered


def get_structures(proto_game, gdf: pd.DataFrame):
    """
    Rename and reorder gdf columns and create structures to continue converting the gdf.
    :param proto_game: protobuf data from analysis object
    :type proto_game: proto
    :param gdf: A dataframe created by an analysis object
    :type gdf: pd.Dataframe
    :return: Lists holding goal-related data, and the modified dataframe.
    :rtype: Union[List[int], List[int], List[str], List[int], pd.Dataframe]
    """
    # Init structures from json
    goal_seconds, goal_frames, goal_scorers, goal_teams = [], [], [], []
    # Player Data
    z_team_names, o_team_names = [], []
    game = proto_game.game_metadata
    for goal in game.goals:
        goal_frames.append(goal.frame_number)
        goal_scorers.append(goal.player_id)  # Currently unused
        if goal.player_id in proto_game.teams[0].player_ids:
            goal_teams.append(0)
        elif goal.player_id in proto_game.teams[1].player_ids:
            goal_teams.append(1)
        else:
            return "Error"
    # Make blue team always team zero
    for player in proto_game.players:
        if proto_game.teams[1].is_orange:
            if player.is_orange:
                o_team_names.append(player.name)
            else:
                z_team_names.append(player.name)
        else:
            if player.is_orange:
                z_team_names.append(player.name)
            else:
                o_team_names.append(player.name)
    # Fix DF columns
    gdf.columns = gdf.columns.to_flat_index()
    # Dictionary for rename
    rename_dict = {}
    for tup in gdf.columns:
        # Need to change all the player values and make them in order
        if tup[0] in z_team_names:
            i = z_team_names.index(tup[0])
            sub = 'z_' + str(i) + '_' + tup[1]

        elif tup[0] in o_team_names:
            i = o_team_names.index(tup[0])
            sub = 'o_' + str(i) + '_' + tup[1]

        else:
            sub = tup[0] + '_' + tup[1]

        rename_dict[tup] = sub

    gdf = gdf.rename(rename_dict, axis='columns')
    # Add demo columns
    for team in ['z_', 'o_']:
        for i in range(NUM_PLAYERS):
            gdf[team + str(i) + '_is_demo'] = np.zeros(len(gdf))

    return goal_seconds, goal_frames, goal_scorers, goal_teams, gdf


def add_game_columns(gdf: pd.DataFrame, goal_frames: List[int], goal_seconds: List[int], goal_teams: List[int]) -> int:
    """
    Mutate the game dataframe, adding columns for goals predictions, and score, and time until goal.
    :param gdf: The game dataframe from the function "replays_to_csv"
    :type gdf: pd.Dataframe
    :param goal_frames:
    :type goal_frames: List[int]
    :param goal_seconds:
    :type goal_seconds: List[int]
    :param goal_teams:
    :type goal_teams: List[int]
    :return: The length of the added columns. Used to truncate the gdf.
    :rtype: int
    """
    goal_one_column = np.empty([0])
    score_0 = np.empty([0])
    score_1 = np.empty([0])
    until_goal = np.array([300])
    score = [0, 0]
    index = 0
    for i in range(len(goal_frames)):
        if i == 0:
            min_index = 0
        else:
            min_index = goal_frames[i - 1]
        # Get the length of the slice of the game between goals
        length = goal_frames[i] - min_index
        index += length
        # Make arrays to be added to columns
        l_1 = np.full(length, goal_teams[i])
        s_0 = np.full(length, score[0])
        s_1 = np.full(length, score[1])

        until_seconds = np.full(length, goal_seconds[i])
        # Get a slice from game_seconds_remaining and get it as an array
        arr_secs_remaining = gdf['game_seconds_remaining'][min_index:goal_frames[i]].to_numpy(copy=True)
        # Concat until we have our full columns
        goal_one_column = np.concatenate((goal_one_column, l_1))
        score_0 = np.concatenate((score_0, s_0))
        score_1 = np.concatenate((score_1, s_1))
        # TODO: Fix how overtime causes negative values (multiple by -ot?)
        until_goal = np.concatenate((until_goal, arr_secs_remaining - until_seconds))
        # Update score
        score[0] += 1 - goal_teams[i]
        score[1] += goal_teams[i]
    # Convert to series so it can be added to the df
    goal_one_column = pd.Series(goal_one_column)
    score_0 = pd.Series(score_0)
    score_1 = pd.Series(score_1)
    until_goal = pd.Series(until_goal)
    # Add columns
    gdf['secs_to_goal'] = until_goal
    gdf['next_goal_one'] = goal_one_column
    gdf['score_zero'] = score_0
    gdf['score_one'] = score_1

    return len(goal_one_column)


# TODO: actually use logging instead of prints
def reporting(shared, interval_mins):
    """
    Multiprocessing function called by pre_process_parallel. Reports the status of the other processes and the work.
    :param shared: A list of multiprocessing values to track errors.
    :type shared: List[Value]
    :param interval_mins: The interval on which to report.
    :type interval_mins: int
    :return: None
    :rtype: None
    """
    try:
        interval = timedelta(minutes=interval_mins)
        in_len = len(os.listdir(replay_path))
        out_len = len(os.listdir(csv_path))
        test_len = len(os.listdir(testcsv_path))
        err_len = len(os.listdir(error_path))
        skip_len = len(os.listdir(skip_path))
        remaining = in_len - out_len - test_len - err_len - skip_len
        print(
            "Starting Reporting\n    There are {} total replays.\n    There are {} replays left to process. ({}%)\n".format(
                in_len, remaining, round(remaining / in_len, 2) * 100))
        sys.stdout.flush()
        # Un-fancy way to ensure the first print is accurate, wait for other processes to start. Fancy method is not much better and more lines.
        time.sleep(10)
        average = [timedelta(0), 0]
        errors = shared[0].value + shared[1].value + shared[2].value + shared[3].value + shared[4].value + shared[
            5].value
        start = datetime.now()
        while remaining > 0:
            try:
                with shared[6].get_lock():
                    if shared[6].value == 0:
                        print("No remaining processes")
                        return
                while (datetime.now() - start) < interval:
                    time.sleep(interval_mins)
                elapsed = datetime.now() - start
                start = datetime.now()

                new_errors = shared[0].value + shared[1].value + shared[2].value + \
                    shared[3].value + shared[4].value + shared[5].value

                processed = (len(os.listdir(csv_path)) - out_len) + (len(os.listdir(testcsv_path)) - test_len) + (
                        new_errors - errors)

                out_len = len(os.listdir(csv_path))
                test_len = len(os.listdir(testcsv_path))
                err_len = len(os.listdir(error_path))
                skip_len = len(os.listdir(skip_path))

                remaining = in_len - out_len - test_len - err_len - skip_len

                errors = new_errors
                if processed == 0:
                    print("Empty Reporting Interval")
                    continue
                average[0] += elapsed
                average[1] += processed

                print("Total errors:{} ({})%".format(errors, round(errors / in_len, 2)))
                print(
                    "Processed {} files.\nThe average time per file is {}".format(processed, (average[0] / average[1])))
                print(
                    "There are {} replays left to process. ({}%)".format(remaining, round(remaining / in_len, 2) * 100))
                print("Total errors:{} ({})%".format(errors, round(errors / in_len, 2)))
                print("Estimated completion in {}\n\n".format(average[0] / average[1] * remaining))
                sys.stdout.flush()

            except KeyboardInterrupt:
                print("Exiting Reporting")
                break

    except KeyboardInterrupt:
        print("Exiting Reporting")

    print(
        "err_analysis_index: {}\nerr_analysis_key: {}\nerr_analysis_rattletrap: {}\nerr_analysis_unbound: {}\nerr_analysis_other: {}\nerr_gdf_index: {}\n".format(
            shared[0].value, shared[1].value, shared[2].value, shared[3].value, shared[4].value, shared[5].value))
    return


def replays_to_csv(in_files: List[str], output_path: str, shared: List[Value]):
    """
    Multiprocessing function called by pre_process_parallel. Converts replay files to csv files.
    :param in_files: A list of files to process.
    :type in_files: List[str]
    :param output_path: The path to save CSVs
    :type output_path: str
    :param shared: A list of multiprocessing values to track errors.
    :type shared: List[Value]
    :return: None
    :rtype: None
    """
    try:
        ordered_cols = get_ordered_columns(NUM_PLAYERS)
        print("Starting {} with {} files".format(current_process().name, len(in_files)))
        file_average = [timedelta(), 0]
        for file in in_files:
            file_start = datetime.now()
            try:
                with HiddenPrints():
                    try:
                        e = False
                        # Analysis has a lot of possible errors
                        analysis = carball.analyze_replay_file(replay_path + file)
                    except IndexError:
                        with shared[0].get_lock():
                            shared[0].value += 1
                        e = True
                        continue
                    except KeyError:
                        with shared[1].get_lock():
                            shared[1].value += 1
                        e = True
                        continue
                    except RattleTrapException:  # Currently can't figure out how to except "carball.rattletrap.run_rattletrap.RattleTrapException"
                        with shared[2].get_lock():
                            shared[2].value += 1
                        e = True
                        continue
                    except UnboundLocalError:
                        with shared[3].get_lock():
                            shared[3].value += 1
                        e = True
                        continue
                    except KeyboardInterrupt:
                        print("Exiting")
                        break
                    except():
                        with shared[4].get_lock():
                            shared[4].value += 1
                        e = True
                        continue
                    finally:
                        if e:
                            copyfile(replay_path + file, error_path + file)
                            print("Copied to err")

                proto_game = analysis.get_protobuf_data()
                gdf = analysis.get_data_frame()
                goal_seconds, goal_frames, goal_scorers, goal_teams, gdf = get_structures(proto_game, gdf)
                # Order the columns of the df
                gdf = gdf[ordered_cols]

                # Determine if the game has overtime
                if (gdf.iloc[-1].game_seconds_remaining == 0) or (0 not in gdf['game_seconds_remaining']):
                    game_has_overtime = False
                else:
                    game_has_overtime = True
                # Get times of each goal
                try:
                    for i in goal_frames:
                        goal_seconds.append(gdf.iloc[i]['game_seconds_remaining'])
                except IndexError:
                    with shared[5].get_lock():
                        shared[5].value += 1
                    copyfile(replay_path + file, error_path + file)
                    print("Error")
                    continue
                # Currently skipping overtimes
                if game_has_overtime:
                    if len(goal_seconds) == 1:
                        copyfile(replay_path + file, skip_path + file)
                        continue
                    else:
                        goal_seconds = goal_seconds[:-1]
                        goal_frames = goal_frames[:-1]
                #
                trunc_length = add_game_columns(gdf, goal_frames, goal_seconds, goal_teams)
                #
                # Fixing up missing values and rows after the last goal
                gdf = gdf.truncate(after=trunc_length - 1)
                # Drop when all player values are NA
                sub = ['z_0_pos_x', 'o_0_pos_x']
                if NUM_PLAYERS > 1:
                    sub.extend(['z_1_pos_x', 'o_1_pos_x'])
                    if NUM_PLAYERS > 2:
                        sub.extend(['z_2_pos_x', 'o_2_pos_x'])
                gdf = gdf.dropna(how='all', subset=sub)
                # forward fill demos (Single player NA), then fill empty values (Ball values)
                for team in ['z_', 'o_']:
                    for i in range(NUM_PLAYERS):
                        num = str(i) + '_'
                        gen_list = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'vel_x', 'vel_y', 'vel_z',
                                    'ang_vel_x', 'ang_vel_y',
                                    'ang_vel_z', 'boost_active', 'jump_active', 'double_jump_active', 'dodge_active']
                        fill_list = [team + num + entry for entry in gen_list]
                        # Change demo column using presence of NA values
                        gdf[team + num + 'is_demo'] = gdf[fill_list].isna().replace({True: 1, False: 0}).mean(axis=1)
                        # Turn NA values into value before demo
                        for _ in fill_list:
                            gdf.loc[:, fill_list] = gdf.loc[:, fill_list].ffill(axis=0)

                # Fill rest of NA value with 0
                gdf = gdf.fillna(0)
                # Drop the time after a goal is scored but before reset
                # (Delete rows where ball velocities are zero but ball x,y is not zero)
                gdf.drop(gdf[(gdf['ball_vel_x'] == 0) & (gdf['ball_vel_y'] == 0) & (gdf['ball_vel_z'] == 0) &
                             (gdf['ball_pos_x'] != 0) & (gdf['ball_pos_y'] != 0) & (gdf['ball_pos_z'] != 0)].index,
                         inplace=True)

                # Change active values to boolean
                gdf['z_0_jump_active'] = ((gdf['z_0_jump_active'] % 2) != 0).astype(int)
                gdf['o_0_jump_active'] = ((gdf['o_0_jump_active'] % 2) != 0).astype(int)
                gdf['z_0_double_jump_active'] = ((gdf['z_0_double_jump_active'] % 2) != 0).astype(int)
                gdf['o_0_double_jump_active'] = ((gdf['o_0_double_jump_active'] % 2) != 0).astype(int)
                gdf['z_0_dodge_active'] = ((gdf['z_0_dodge_active'] % 2) != 0).astype(int)
                gdf['o_0_dodge_active'] = ((gdf['o_0_dodge_active'] % 2) != 0).astype(int)
                # TODO: Check if anything needs to be filled with a non-zero value?
                # Convert all booleans to 0 or 1
                gdf = gdf.replace({True: 1, False: 0})
                # Now we need to handle the duplicates from kickoffs
                gdf = gdf.drop_duplicates()
                # Reduce size in memory by ~half
                # (Not handling booleans differently ex: player_jump_active: True -> 1 -> 1.0)
                gdf = gdf.astype('float32')
                # Write out to CSV
                gdf.to_csv(output_path + file.split('.')[0] + '.csv')
                file_average[1] += 1
                file_average[0] += (datetime.now() - file_start)
                sys.stdout.flush()
            except(KeyboardInterrupt, SystemExit):
                break
    except(KeyboardInterrupt, SystemExit):
        pass
    # Update number of processes so reporting doesn't idle
    with shared[6].get_lock():
        shared[6].value -= 1
    print("{} Exiting".format(current_process().name))
    return


def pre_process_parallel(num_processes, test_ratio=.1, overwrite=False, verbose_interval=10):
    """
    Execute a number of processes to pre-process replay files to CSVs, as designated by the config.
    :param num_processes: The number of processes to run.
    :type num_processes: int
    :param test_ratio: The ratio of replays to use as a test set
    :type test_ratio: float
    :param overwrite: Whether to overwrite existing CSVs
    :type overwrite: bool
    :param verbose_interval: How often to print information (Minutes)
    :type verbose_interval: float
    :return: None
    :rtype: None
    """
    # check if num_processes is reasonable
    if num_processes < 1:
        print("Processes must be at least 1")
        return
    if num_processes > mp.cpu_count():
        print("Running more processes than cpu_count")
    # prepare paths
    for p in [csv_path, testcsv_path, error_path, skip_path]:
        if not os.path.exists(p):
            os.makedirs(p)
            print("Created directories in {}".format(p))

    # Get file names
    in_files = os.listdir(replay_path)
    out_files = os.listdir(csv_path)
    out_test = os.listdir(testcsv_path)
    err_files = os.listdir(error_path)
    skip_files = os.listdir(skip_path)
    # Tracking errors and handling separate test output
    total = len(in_files)
    # TODO: Restructure these loops into 1?
    # Skip existing CSVs unless we are overwriting, and count extraneous CSVs
    extraneous = 0
    if not overwrite:
        for file in out_files:
            if file.split('.')[0] + '.replay' in in_files:
                in_files.remove(file.split('.')[0] + '.replay')
            else:
                extraneous += 1
        for file in out_test:
            if file.split('.')[0] + '.replay' in in_files:
                in_files.remove(file.split('.')[0] + '.replay')
            else:
                extraneous += 1
    # Skip error replays
    err_count = 0
    for file in err_files + skip_files:
        if file in in_files:
            in_files.remove(file)
            err_count += 1

    print("Skipping {} files recorded as causing errors or meeting criteria to be skipped.".format(err_count))
    print("There are {} existing output CSV's that don't correspond with a replay file.".format(extraneous))
    if len(in_files) == 0:
        print("No replays left to process")
        return
    # Remove duplicate CSV's in test and out (from out)
    duplicates = 0
    for file in out_test:
        if file in out_files:
            duplicates += 1
            os.remove(csv_path + file)

    # Tracking total counts of things
    err_analysis_index = Value('I', 0)
    err_analysis_key = Value('I', 0)
    err_analysis_rattletrap = Value('I', 0)
    err_analysis_unbound = Value('I', 0)
    err_analysis_other = Value('I', 0)
    err_gdf_index = Value('I', 0)
    running_processes = Value('I', 0)

    shared = [err_analysis_index, err_analysis_key, err_analysis_rattletrap,
              err_analysis_unbound, err_analysis_other, err_gdf_index, running_processes]
    processes: List[Process] = []
    if verbose_interval > 0:
        reporter = Process(target=reporting, args=(shared, verbose_interval))
    # Randomly get test set
    random.shuffle(in_files)
    if test_ratio is not 0:
        total_test = int(test_ratio * total)
        num_test = total_test - len(out_test)
        if num_test > len(in_files):
            print(
                "There are not enough input files to produce the desired test ratio. This can happen if overwrite = False and we have already partially processed these folders with a smaller ratio.")
            return
        if num_test > 0:
            test_files = in_files[:num_test]
            in_files = in_files[num_test:]
            processes.append(Process(target=replays_to_csv, args=(test_files, testcsv_path, shared)))
            # If only 1 process (why?), just do it all and return (Pretend you didn't see this code)
            num_processes -= 1
            if num_processes == 0:
                processes[0].start()
                processes[0].join()
                process = Process(target=replays_to_csv, args=(in_files, csv_path, shared))
                process.start()
                process.join()
                return
        else:
            print("There are already more files in the test set than given by the ratio.")
    # TODO: Stop bottlenecking on the test set process

    # Split up the work and start processes
    workloads = np.array_split(in_files, num_processes)
    if verbose_interval > 0:
        reporter.start()
        # Sleep just so that reporter prints more accurately :)
        time.sleep(1.5)
    for i in range(num_processes):
        processes.append(Process(target=replays_to_csv, args=(workloads[i], csv_path, shared)))
    for p in processes:
        p.start()
        with shared[6].get_lock():
            shared[6].value += 1
    for p in processes:
        p.join()
    if verbose_interval > 0:
        reporter.join()
    return
