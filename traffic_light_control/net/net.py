
from net.single_intersection import SingleIntesection


def get_net(id_: str, config):
    if id_ == "single_intersection":
        return SingleIntesection(
            input_space=config["input_space"],
            output_space=config["output_space"])
    return None
