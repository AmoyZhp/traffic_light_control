import json
import torch
from hprl.recorder.recorder import Recorder


class TorchReader(Recorder):
    def __init__(self) -> None:
        super().__init__()

    def read_config(self, dir: str, filename: str):
        raise NotImplementedError

    def read_ckpt(self, file_path: str, dir=""):
        return torch.load(file_path)

    def read_records(self, dir: str, filename: str):
        raise NotImplementedError

    def add_record(self, record=None):
        ...

    def add_records(self, records=None):
        ...

    def print_record(
        self,
        record=None,
        logger=None,
        fig=None,
    ):
        ...

    def write_records(self, dir: str = "", filename: str = ""):
        ...

    def write_ckpt(self, ckpt=None, dir: str = "", filename: str = ""):
        ...

    def write_config(self, config=None, dir: str = "", filename: str = ""):
        ...

    def get_records(self):
        ...