from policy.buffer.basic_buffer import BasicBuffer
from policy.buffer.on_policy_buffer import OnPolicyBuffer


def get_buffer(id_, config):
    if id_ == "basis":
        return __get_basis_buffer(config)
    if id_ == "on_policy_buffer":
        return OnPolicyBuffer(config["capacity"])
    else:
        print("invalid buffer id {}".format(id_))


def __get_basis_buffer(config):
    return BasicBuffer(config["capacity"])
