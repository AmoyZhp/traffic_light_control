from policy.buffer.basic_buffer import BasicBuffer


def get_buffer(id_, config):
    if id_ == "basis":
        return __get_basis_buffer(config)
    else:
        print("invalid buffer id {}".format(id_))


def __get_basis_buffer(config):
    return BasicBuffer(config["capacity"])
