import trainer
import cityflow


def static_test():
    path = "config/syn_1x3_gaussian_500_1h/config.json"
    max_time = 360
    eng = cityflow.Engine(path, thread_num=1)
    for t in range(max_time):
        eng.next_step()


def rl_train():
    id_ = "independent"
    tr = trainer.get_trainer(id_, {})
    tr.run()


if __name__ == "__main__":
    rl_train()
