from data.download_replays import download_replays_range
from data.convert_replays import pre_process_parallel
from data.create_dataset import dataset
import sys


def main():
    nargs = len(sys.argv)
    if nargs > 1:
        # Downloading Replays
        if sys.argv[1] == 'download':
            if nargs == 2:
                download_replays_range()
            if nargs == 3:
                download_replays_range(max_downloaded=int(sys.argv[2]))
            else:
                print("download usage: download [max_downloaded]")
        # Converting Replays to CSV
        if sys.argv[1] == 'convert':
            if nargs == 5:
                pre_process_parallel(int(sys.argv[2]), test_ratio=float(sys.argv[3]), overwrite=False,
                                     verbose_interval=float(sys.argv[4]))
            else:
                print("convert usage: convert <num_processes> <test_ratio>  <verbose_interval>")
                print("\nNotes: test_ratio and verbose_interval are floats which can be 0 if you don't want either functionality.")
                print("test_ratio = .1 means 10% of the replays will go into the test folder. A single process handles test, so if the ratio is >>1/num_processes it will bottleneck.")
                print("verbose_interval = 10 means print every 10 minutes. ")
        # Creating dataset from csv
        if sys.argv[1] == 'dataset':
            if nargs == 5:
                output = None
            elif nargs == 6:
                output = sys.argv[5]
            else:
                print("dataset usage: dataset <0 (No test set) | 1 (with test set) | 2 (only test set)> <max_games | 0> <chunk_size | 0> [output_filename]")
                print("Each csv is a single game. Chunk size refers to rows/frames, not games. Actual chunk_size is always >= this param (it overshoots)")
                print("output_filename overrides the default naming.")
                print("ex: dataset 1 1000 250000")
                return
            if int(sys.argv[3]) == 0:
                mg = None
            else:
                mg = int(sys.argv[3])

            if int(sys.argv[4]) == 0:
                cs = None
            else:
                cs = int(sys.argv[4])
            t_int = int(sys.argv[2])
            if t_int >= 1:
                dataset(output_file=output, test=True, max_games=mg, chunk_size=cs)
            if t_int <= 1:
                dataset(output_file=output, test=False, max_games=mg, chunk_size=cs)

    else:
        print("Usage: download | convert | dataset")
        print("download: download [max_replays]")
        print("convert: convert <num_processes> <test_ratio>  <verbose_interval>")
        print(
            "dataset: dataset <0 (No test set) | 1 (with test set) | 2 (only test set)> <max_games | 0> <chunk_size | 0>")


if __name__ == "__main__":
    main()
