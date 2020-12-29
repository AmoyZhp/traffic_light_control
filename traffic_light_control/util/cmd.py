import argparse


DATA_SAVE_PERIOD = 100
SAVED_THRESHOLD = -100.0

ENV_ID = "hangzhou_1x1_bc-tyc_18041607_1h"


def parase_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", "--mode", type=str, required=True,
        help="mode of exec, include [train, test, static]")

    parser.add_argument(
        "-dsp", "--data_saved_period", type=int,
        required=True,
        help="saving period of data"
    )

    parser.add_argument(
        "-st", "--saved_threshold", type=float,
        required=True,
        help="the threhold that save model weight"
    )

    parser.add_argument(
        "-env", "--environment", type=str,
        required=True,
        help="the id of environment"
    )

    parser.add_argument(
        "-alg", "--algorithm", type=str,
        required=True,
        help="which algorithm be chosen"
    )

    parser.add_argument(
        "-wrap", "--wrapper", type=str,
        required=True,
        help="the wrapper of algorithm"
    )

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

    return parser.parse_args()
