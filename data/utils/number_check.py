from typing import List

import requests

BASE_URL = 'https://calculated.gg/api/v1/'
NUM = 200


def get_query(playlist: int = 11):
    return BASE_URL + f'replays?key=1&minmmr={MIN_MMR}&playlist={playlist}&num={NUM}&page=0&year=2018'


def get_replay_list(query: str) -> List[str]:
    r = requests.get(query)
    return [replay['hash'] for replay in r.json()['data']]


def check_playlists():
    PLAYLISTS_TO_TEST = [
        1, 2, 3, 4, 6, 8, 10, 11, 12, 13, 15, 16, 27, 28, 29, 30
    ]

    playlist_count = {}

    for playlist in PLAYLISTS_TO_TEST:
        query = get_query(playlist)
        print(f"query: {query}")

        replay_ids = set()

        current_length = 0
        i = 0
        while True:
            print(f"current count: {current_length}")
            _replay_ids = get_replay_list(query)
            replay_ids.update(_replay_ids)

            new_length = len(replay_ids)
            print(f"new: {new_length - current_length}")
            if new_length == current_length:
                i += 1
                if i > 3:
                    print(f"found total of {new_length} replays for playlist {playlist}")
                    playlist_count[playlist] = new_length
                    break

            current_length = new_length

    print(playlist_count)


def check_playlist(playlist: int):
    query = get_query(playlist)
    print(f"query: {query}")

    replay_ids = set()

    current_length = 0
    i = 0
    while True:
        print(f"current count: {current_length}")
        _replay_ids = get_replay_list(query)
        replay_ids.update(_replay_ids)

        new_length = len(replay_ids)
        print(f"new: {new_length - current_length}")
        if new_length == current_length:
            i += 1
            if i > 3:
                print(f"found total of {new_length} replays for playlist {playlist}")
                break

        current_length = new_length


if __name__ == '__main__':
    # MIN_MMR = 1500
    # check_playlists()
    MIN_MMR = 1300
    check_playlist(13)
