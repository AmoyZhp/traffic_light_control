from trainer.independent_traffic_trainer import IndependentTrainer


def get_trainer(id_: str, config):
    if id_ == "independent":
        return __get_independent_exectuor(config)
    return None


def __get_independent_exectuor(config):
    return IndependentTrainer()
