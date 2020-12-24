from buffer.replay_buffer import ReplayBuffer


def get_buffer(id_, config):
    if id_ == "basis":
        return __get_basis_buffer(config)
    else:
        print("invalid buffer id {}".format(id_))


def __get_basis_buffer(config):
    return ReplayBuffer(config["capacity"])
