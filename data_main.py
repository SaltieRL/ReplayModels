from data.download_replays import download_replays_range
from data.convert_replays import pre_process_parallel
import sys
if __name__ == "__main__":
    nargs = len(sys.argv)
    if nargs > 1:
        if sys.argv[1] == 'download':
            if nargs == 2:
                download_replays_range()
            if nargs == 3:
                download_replays_range(max_downloaded = int(sys.argv[2]))
            print("download usage: download [max_downloaded]")
        if sys.argv[1] == 'convert':
            if nargs == 5:
                pre_process_parallel(int(sys.argv[2]), test_ratio = float(sys.argv[3]), overwrite = False, verbose_interval = float(sys.argv[4]))
            else:
                print("Convert usage: convert <num_processes> <test_ratio>  <verbose_interval>")
                print("\nNotes: test_ratio and verbose_interval are floats which can be 0 if you don't want either functionality. verbose_interval = 10 means print every 10 minutes. ")


    else:
        print("Usage: download | convert | TODO: dataset")
        print("download: download [max ]")
        print("Convert: convert <num_processes> <test_ratio>  <verbose_interval>")
    