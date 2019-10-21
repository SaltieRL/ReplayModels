from data.interfacers.calculatedgg_api.api_interfacer import CalculatedApiInterfacer
from data.interfacers.calculatedgg_api.query_params import CalculatedApiQueryParams
import requests
import os
from configparser import ConfigParser, ExtendedInterpolation
import pandas as pd

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('data/config.ini')
mode_tuple = config['VARS']['MODE'].split(',')
mmrs = config['VARS']['mmr_range'].split('-')
paths = config['PATHS']


def download_replays_range(playlist_tuple=mode_tuple, min_mmr=mmrs[0], max_mmr=mmrs[1],
                           replay_path=paths['replay_path'],
                           replay_log=paths['replay_log'], max_downloaded=None):
    """
    Download replays from calculated.gg to the replay folder. Uses the config.
    :param replay_log: Path to logging csv
    :type replay_log: str
    :param playlist_tuple: Which playlist to draw from.
    :type playlist_tuple: tuple(int,int)
    :param min_mmr:
    :type min_mmr: int
    :param max_mmr:
    :type max_mmr: int
    :param replay_path: Path to replays folder
    :type replay_path: str
    :param max_downloaded: Maximum replays to download.
    :type max_downloaded: Optional[int]
    :return: None
    :rtype: None
    """
    if not os.path.exists(replay_path):
        os.makedirs(replay_path)
        print("Created directories in replay_path")
    mode = int(playlist_tuple[0])
    num_players = int(playlist_tuple[1])
    data_cols = ['hash', 'download', 'map', 'match_date', 'upload_date', 'team_blue_score', 'team_orange_score'] + [
        f'p_{x}' for x in range(num_players * 2)] + [f'mmr_{x}' for x in range(num_players * 2)]
    if not os.path.exists(replay_log):
        log = pd.DataFrame(columns=data_cols)
        log.to_csv(replay_log)
        print("Created replay log")

    params = CalculatedApiQueryParams(1, 200, mode, min_mmr, max_mmr)
    interfacer = CalculatedApiInterfacer(params)
    if max_downloaded is not None:
        pages = (max_downloaded // 200) + 1
        last_page_len = max_downloaded % 200
    else:
        pages = 100
        last_page_len = 0
    existing = os.listdir(replay_path)
    log = pd.read_csv(replay_log, index_col=0)
    logged = log['hash'].values
    old = 0
    new = 0
    logs = []

    for page in range(pages):
        if page == 0:
            page_req = interfacer._get_replays_request(params).json()
            print(f"Matched {page_req['total_count']} replays")
            page_data = page_req['data']
        else:
            page_data = interfacer._get_replays_request(params).json()['data']
        if page == pages - 1:
            page_data = page_data[:last_page_len]
        if len(page_data) == 0:
            break
        for game in page_data:
            # Skip existing
            name = game['hash']
            if name + '.replay' not in existing:
                # Write replay file
                open(replay_path + name + '.replay', 'wb').write(
                    requests.get(f"https://calculated.gg/api/replay/{name}/download").content)
                new += 1
            else:
                old += 1

            if name not in logged:
                # Logging
                log_row = {x: None for x in data_cols}
                for key in list(set(log_row.keys()) & set(game.keys())):
                    log_row[key] = game[key]
                for i in range(len(game['players'])):
                    log_row[f'p_{i}'] = game['players'][i]
                # MMRS has unknown length, missing MMRS will be 'None'
                for i in range(len(game['mmrs'])):
                    log_row[f'mmr_{i}'] = game['mmrs'][i]

                logs.append(log_row)

        params = params._replace(page=params.page + 1)

    new_log = pd.DataFrame(logs, columns=data_cols)
    pd.concat([log, new_log]).to_csv(replay_log)
    print(f"Existing: {old}")
    print(f"Wrote new: {new}")
    print(f"Logged existing: {len(new_log) - new}")
