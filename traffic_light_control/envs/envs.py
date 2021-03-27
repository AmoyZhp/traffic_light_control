import envs.cityflow as mycityflow


def make_cityflow(config):
    return mycityflow.make(config)


def make_mp_cityflow(config):

    return mycityflow.make_mp(config)
