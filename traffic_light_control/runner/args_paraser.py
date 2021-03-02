import argparse
import hprl


DEFAULT_BATCH_SIZE = 16


def args_validity_check(args):
    mode = args.mode
    if mode not in ["train", "test"]:
        print(" model value invalid !")
        return False

    if mode == "test":
        if args.ckpt_file is None:
            print("checkpoint file should not be none"
                  "if want to test")
            return False
        if args.record_dir is None:
            print("record dir should not be none"
                  "if want to test")
    trainer = hprl.TrainnerTypes(args.trainer)
    if trainer not in hprl.TrainnerTypes:
        print("trainer type is invalid !")
        return False
    if args.resume:
        if args.ckpt_file is None:
            print("checkpoint file should not be none"
                  "if want to resume training")
            return False
        if args.record_dir is None:
            print("record dir should not be none"
                  "if want to resume training")
            return False
    return True


def create_paraser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", type=str, required=True,
        help="Mode of execution. Include [ train , test ]"
    )

    parser.add_argument(
        "--env", type=str, required=True,
        help="the id of the environment"
    )

    parser.add_argument(
        "--trainer", type=str, required=True,
        help="which trainer to be chosen"
    )

    parser.add_argument(
        "--episodes", type=int, default=0,
        help="episode of train times"
    )

    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help="batch size of sample batch"
    )

    parser.add_argument(
        "--replay_buffer", type=str, default="Common",
        help="type of replay buffer chosen "
    )

    parser.add_argument(
        "--eval_episodes", type=int, default=1,
        help="episode of eval times"
    )

    parser.add_argument(
        "--eval_frequency", type=int, default=0,
        help="eval frequency in training process"
    )

    parser.add_argument(
        "--check_frequency", type=int, default=0,
        help="the frequency of saving checkpoint."
        "if it less than or equal zero, checkpoint would not be saved"
    )

    parser.add_argument(
        "--env_thread_num", type=int, default=1,
        help="thread number of env"
    )

    parser.add_argument(
        "--save_replay", action="store_true",
        help="whether cityflow env save replay or not"
    )

    parser.add_argument(
        "--resume", action="store_true",
        help="resume training or not"
    )

    parser.add_argument(
        "--record_dir", type=str, default=None,
        help="directory of record"
    )
    parser.add_argument(
        "--ckpt_file", type=str, default=None,
        help="name of checkpoint file"
    )

    return parser
