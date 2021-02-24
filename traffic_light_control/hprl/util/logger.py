
class Logger(object):
    def __init__(self,
                 log_dir: str) -> None:
        self._log_dir = log_dir
        self.data = {}

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]
