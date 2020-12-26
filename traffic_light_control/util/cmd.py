import argparse


DATA_SAVE_PERIOD = 100
SAVED_THRESHOLD = -100.0


def parase_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mode", type=str, default="", required=True,
        help="mode of exec, include [train, test, static]")
    parser.add_argument(
        "-e", "--episodes", type=int, default=1,
        help="episode of exectue time"
    )
    parser.add_argument(
        "-mf", "--model_file", type=str,
        help="the path of model parameter file"
    )
    parser.add_argument(
        "-th", "--thread_num", type=int, default=1,
        help="thread number of simulator"
    )
    parser.add_argument(
        "-s", "--save", type=int, default=1,
        help="save params or not"
    )
    parser.add_argument(
        "-r", "--resume", type=int, default=0,
        help="resume training or not"
    )
    parser.add_argument(
        "-rd", "--record_dir", type=str, default="",
        help="resume dir if set resume with true"
    )

    parser.add_argument(
        "-dsp", "--data_saved_period", type=int, default=DATA_SAVE_PERIOD,
        required=True,
        help="saving period of data"
    )

    parser.add_argument(
        "-st", "--saved_threshold", type=float, default=SAVED_THRESHOLD,
        required=True,
        help="the threhold that save model weight"
    )

    return parser.parse_args()
