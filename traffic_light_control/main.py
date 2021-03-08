import hprl
import runner
import cityflow


def main():
    runner.run()


def static_test():
    path = "cityflow_config/{}/config.json".format("LA_1x4")
    max_time = 360
    eng = cityflow.Engine(path, thread_num=1)
    eng.set_save_replay(True)
    for t in range(max_time):
        eng.next_step()


if __name__ == "__main__":
    main()