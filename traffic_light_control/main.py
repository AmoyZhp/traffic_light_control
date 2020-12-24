import trainer
import cityflow


def static_test():
    path = "config/single_complete_1x1_static/config.json"
    max_time = 60
    eng = cityflow.Engine(path, thread_num=1)
    for t in range(max_time):
        eng.next_step()
        print("====== time {} ======".format(t))
        print(eng.get_lane_vehicle_count())


def rl_train():
    id_ = "independent"
    tr = trainer.get_trainer(id_, {})
    tr.run()


if __name__ == "__main__":
    rl_train()
