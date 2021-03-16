import argparse
import hprl

# Agent Setting
CAPACITY = 16384
CRITIC_LR = 1e-4
ACTOR_LR = 1e-4
DISCOUNT_FACTOR = 0.99
EPS_INIT = 1.0
EPS_MIN = 0.01
EPS_FRAME = 300000
UPDATE_PERIOD = 1000
INNER_EPOCH = 32
CLIP_PARAM = 0.2
PER_ALPHA = 0.6
PER_BETA = 0.4


def args_validity_check(args):
    mode = args.mode
    if mode not in ["train", "test"]:
        print(" model value invalid !")
        return False

    if mode == "test":
        if args.ckpt_file is None:
            print("checkpoint file should not be none" "if want to test")
            return False
        if args.record_dir is None:
            print("record dir should not be none" "if want to test")
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
            print("record dir should not be none" "if want to resume training")
            return False
    return True


def create_paraser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",
                        type=str,
                        default="train",
                        help="Mode of execution. Include [ train , test ]")

    parser.add_argument("--env",
                        type=str,
                        required=True,
                        help="the id of the environment")

    parser.add_argument("--trainer",
                        type=hprl.TrainnerTypes,
                        required=True,
                        help="which trainer to be chosen")

    parser.add_argument("--episodes",
                        type=int,
                        default=0,
                        help="episode of train times")

    parser.add_argument("--batch_size",
                        type=int,
                        required=True,
                        help="batch size of sample batch")

    parser.add_argument("--replay_buffer",
                        type=hprl.ReplayBufferTypes,
                        default="Common",
                        help="type of replay buffer chosen ")

    parser.add_argument("--eval_episodes",
                        type=int,
                        default=1,
                        help="episode of eval times")

    parser.add_argument("--eval_frequency",
                        type=int,
                        default=0,
                        help="eval frequency in training process")

    parser.add_argument(
        "--ckpt_frequency",
        type=int,
        default=0,
        help="the frequency of saving checkpoint."
        "if it less than or equal zero, checkpoint would not be saved")

    parser.add_argument("--env_thread_num",
                        type=int,
                        default=1,
                        help="thread number of env")

    parser.add_argument("--save_replay",
                        action="store_true",
                        help="whether cityflow env save replay or not")

    parser.add_argument("--resume",
                        action="store_true",
                        help="resume training or not")

    parser.add_argument("--record_dir",
                        type=str,
                        default=None,
                        help="directory of record")
    parser.add_argument("--ckpt_file",
                        type=str,
                        default=None,
                        help="name of checkpoint file")
    parser.add_argument(
        "--update_period",
        type=int,
        default=UPDATE_PERIOD,
        help="update peirod of dqn",
    )

    parser.add_argument(
        "--inner_epoch",
        type=int,
        default=INNER_EPOCH,
        help="inner train epoch of ppo",
    )

    parser.add_argument(
        "--critic_lr",
        type=float,
        default=CRITIC_LR,
        help="learning rate of critic net work",
    )

    parser.add_argument(
        "--actor_lr",
        type=float,
        default=ACTOR_LR,
        help="learning rate of actor",
    )

    parser.add_argument(
        "--capacity",
        type=int,
        default=CAPACITY,
        help="capacity of replay buffer",
    )
    parser.add_argument(
        "--clip_param",
        type=float,
        default=CLIP_PARAM,
        help="clip param of ppo",
    )
    parser.add_argument(
        "--eps_frame",
        type=int,
        default=EPS_FRAME,
        help="framed has be taken to eps min",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DISCOUNT_FACTOR,
        help="discount factor of rl",
    )
    parser.add_argument(
        "--per_beta",
        type=float,
        default=PER_BETA,
        help="beta used in prioritized exp replay",
    )
    parser.add_argument(
        "--per_alpha",
        type=float,
        default=PER_ALPHA,
        help="alpha value used in prioritized exp replay",
    )
    parser.add_argument("--recording",
                        action="store_false",
                        help="recording or not")

    return parser
