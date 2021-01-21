import argparse


BATCH_SIZE = 256


def parase_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", "--mode", type=str, required=True,
        help="mode of exec, include [train, test, static]")

    parser.add_argument(
        "-env", "--environment", type=str,
        required=True,
        help="the id of environment"
    )

    parser.add_argument(
        "-ply", "--policy", type=str,
        required=True,
        help="which policy algorithm be chosen"
    )

    parser.add_argument(
        "-e", "--episodes", type=int, default=1,
        help="episode of exectue time"
    )

    parser.add_argument(
        "-dsp", "--data_saved_period", type=int, default=1000000,
        help="saving period of data"
    )

    parser.add_argument(
        "-st", "--saved_threshold", type=float, default=-1,
        help="the threhold that save model weight"
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
        "-bs", "--batch_size", type=int, default=BATCH_SIZE,
        help="batch size"
    )

    return parser.parse_args()
