
from policy.net.nets import IActor, SingleIntesection


def get_net(id_: str, config):
    if id_ == "single_intersection":
        return SingleIntesection(
            input_space=config["input_space"],
            output_space=config["output_space"])
    elif id_ == "IActor":
        return IActor(
            input_space=config["input_space"],
            output_space=config["output_space"]
        )
    return None
