from exec.exectuor import Exectutor
import sys
import getopt

if __name__ == "__main__":
    executor = Exectutor()
    argv = sys.argv[1:]
    mode = ""
    path = ""
    thread = 1
    episode = 1
    opts = []
    try:
        opts, args = getopt.getopt(
            argv, shortopts=["m:e:p:th"],
            longopts=["mode=",
                      "episode=", "path=", "thread="])
        for opt, arg in opts:
            if opt in ("-m", "--mode"):
                mode = arg
            elif opt in ("-p", "--path"):
                path = arg
            elif opt in ("-e", "--episode"):
                episode = arg
            elif opt in ("-th", "--thread"):
                thread = int(arg)
        print("mode {}, path {}, ep {}, thread {}".format(
            mode, path, episode, thread
        ))

        if mode == "test":
            executor.test(path)
        elif mode == "train":
            executor.train(episode=episode, thread_num=thread)
        else:
            print("please input mode")
    except getopt.GetoptError:
        print("python test.py --mode=train"
              + " --episode=1 --path=model.pth --thread=1")
