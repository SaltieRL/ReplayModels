from typing import List

import requests

BASE_URL = 'https://calculated.gg/api/v1/'
NUM = 200


def get_query(playlist: int = 11):
    return BASE_URL + f'replays?key=1&minmmr={MIN_MMR}&playlist={playlist}&num={NUM}&page=0&year=2018'


def get_replay_list(query: str) -> List[str]:
    """
    Does query and parses response for replay ids.
    :param query: query (given from get_query())
    :return: List of replay ids
    """
    r = requests.get(query)
    return [replay['hash'] for replay in r.json()['data']]


def check_playlists():
    """
    Checks query for a list of playlists. (see PLAYLISTS_TO_TEST variable below).
    :return: None
    """
    PLAYLISTS_TO_TEST = [
        1, 2, 3, 4, 6, 8, 10, 11, 12, 13, 15, 16, 27, 28, 29, 30
    ]

    playlist_count = {}

    for playlist in PLAYLISTS_TO_TEST:
        count = check_playlist(playlist)
        playlist_count[playlist] = count

    print(playlist_count)


def check_playlist(playlist: int) -> int:
    """
    Checks available replays for the given playlist
    :param playlist: the playlist query param
    :return: The number of available replays for the given playlist.
    """
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
    return new_length


if __name__ == '__main__':
    # MIN_MMR = 1500
    # check_playlists()
    MIN_MMR = 1500
    check_playlist(13)
