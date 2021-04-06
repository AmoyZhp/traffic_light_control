import hprl
import controler
import cityflow


def main():
    controler.run()


def static_test():
    path = "cityflow_config/{}/config.json".format(
        "hangzhou_1x1_bc-tyc_18041607_1h")
    max_time = 360
    eng = cityflow.Engine(path, thread_num=1)
    eng.set_save_replay(True)
    for t in range(max_time):
        eng.next_step()


if __name__ == "__main__":
    main()