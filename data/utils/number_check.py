from typing import List

import requests

from data.utils.playlists import Playlist

BASE_URL = 'https://calculated.gg/api/v1/'


def get_query(playlist: int = 11):
    return BASE_URL + f'replays?key=1&minmmr={MIN_MMR}&playlist={playlist}&year=2019'


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
    # PLAYLISTS_TO_TEST = [
    #     1, 2, 3, 4, 6, 8, 10, 11, 12, 13, 15, 16, 27, 28, 29, 30
    # ]

    playlist_count = {}

    for playlist in Playlist:
        count = check_playlist(playlist.value)
        playlist_count[playlist.name] = count

    print(playlist_count)


def check_playlist(playlist: int) -> int:
    """
    Checks available replays for the given playlist
    :param playlist: the playlist query param
    :return: The number of available replays for the given playlist.
    """

    query = get_query(playlist)
    print(f"query: {query}")
    r = requests.get(query)

    count = r.json()['total_count']
    print(f"count: {count}")
    return count


if __name__ == '__main__':
    # MIN_MMR = 1500
    # check_playlists()
    MIN_MMR = 1700
    check_playlist(13)  # Ranked Standard
    # MIN_MMR = 1300
    # check_playlist(10)  # Ranked Duels
