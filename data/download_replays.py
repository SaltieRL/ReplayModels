from data.interfacers.calculatedgg_api.api_interfacer import CalculatedApiInterfacer
from data.interfacers.calculatedgg_api.query_params import CalculatedApiQueryParams
import requests
import os
from configparser import ConfigParser, ExtendedInterpolation
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('data/config.ini')
mode = config['VARS']['MODE'].split(',')
mmrs = config['VARS']['mmr_range'].split('-')
paths = config['PATHS']

def download_replays_range(playlist_int = mode[0], min_mmr = mmrs[0], max_mmr = mmrs[1], replay_path = paths['replay_path'], max_downloaded = None):
    '''playlist_int (int)- Which playlist to download
     min_mmr, max_mmr (ints)- The range from the playlist to download 
     max_downloaded, max_replays (ints | None & int)- How many to download, total max we want in the directory
     replay_path - path to download to'''
    if not os.path.exists(replay_path):
        os.makedirs(replay_path)
        print("Created directories in replay_path")

    d_urls = ['https://calculated.gg/api/replay/', '/download']
    params = CalculatedApiQueryParams(1, 200, playlist_int, min_mmr, max_mmr)
    interfacer = CalculatedApiInterfacer(params)    
    full_list = list(interfacer.get_all_replay_ids())
    print("Found {} replays.".format(len(full_list)))
    
    existing = os.listdir(replay_path)
    old = len(existing)
    new = 0
    for i in range(len(full_list)):
        if (max_downloaded is not None) and (new >= max_downloaded):
            print("Reached maximum downloaded replays. ({})".format(max_downloaded))
            break
        rid = full_list[i]
        if rid + '.replay' in existing:
            old += 1
        else:
            new +=1
            replay_file = requests.get('https://calculated.gg/api/replay/{}/download'.format(rid))
            open(replay_path + rid + '.replay', 'wb').write(replay_file.content)
        
    print("Existing: {}".format(old))
    print("Wrote new: {}".format(new))