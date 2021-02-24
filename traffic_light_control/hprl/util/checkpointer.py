import os
from typing import Dict
import torch


class Checkpointer(object):
    def __init__(self,
                 base_directory: str,
                 checkpoint_frequency: int = 1,
                 checkpoint_file_prefix: str = "ckpt") -> None:

        if not base_directory:
            raise ValueError('No path provided to Checkpointer.')

        self.base_directory = base_directory
        if not os.path.exists(base_directory):
            os.mkdir(base_directory)

        self.checkpoint_file_prefix = checkpoint_file_prefix
        self.checkpoint_frequency = checkpoint_frequency

    def periodic_save(self, data, iteration: int):
        if iteration % self.checkpoint_frequency != 0:
            return
        filename = f'{self.checkpoint_file_prefix}_{iteration}.pth'
        self.save(data, filename)

    def quick_load(self, iteration: int):
        filename = f'{self.checkpoint_file_prefix}_{iteration}.pth'
        return self.load(filename)

    def save(self, data, filename: str):
        filename = f'{self.base_directory}/{filename}'
        torch.save(data, filename)

    def load(self, filename: str) -> Dict:
        filename = f'{self.base_directory}/{filename}'
        data = torch.load(filename)
        return data
