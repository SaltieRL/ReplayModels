from enum import Enum


class Playlist(Enum):
    UNRANKED_SOLO_DUEL = 1
    UNRANKED_DOUBLES = 2
    UNRANKED_STANDARD = 3
    UNRANKED_CHAOS = 4
    RANKED_DUEL = 10
    RANKED_DOUBLES = 11
    RANKED_SOLO_STANDARD = 12
    RANKED_STANDARD = 13
    RANKED_HOOPS = 27
    RANKED_RUMBLE = 28
    RANKED_DROPSHOT = 29
    RANKED_SNOW_DAY = 30