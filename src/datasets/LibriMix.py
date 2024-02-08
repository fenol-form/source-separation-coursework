from typing import Tuple

import torch
import os

from src.datasets.base import BaseDataset
from asteroid.data import LibriMix


class WorkingDirectoryChanger:

    def __init__(self, new_dir=""):
        self.cur_directory = os.getcwd()
        self.new_directory = new_dir

    def __enter__(self):
        os.chdir(self.new_directory)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.cur_directory)


class LibriMixDataset(BaseDataset):
    """
    Asteroid LibriMix wrapper
    """
    def __init__(self, datasetDir=None, task="sep_clean", **kwargs):
        super().__init__(**kwargs)
        assert self.mode in ["train", "val"], "mode should be either 'train' or 'val'"

        if datasetDir is None:
            self.train_set, _ = LibriMix.mini_from_download(task=task)
        else:
            # some workaround (hydra's outputs dir doesn't allow to access LibriMix directly)
            # TODO: change
            with WorkingDirectoryChanger("../../../") as wd:
                self.train_set = LibriMix(datasetDir, task)

    def __getitem__(self, idx) -> dict:
        """
        :param idx: int
        :return: element in form of (source mixture, target: Tuple[Tensor])
        """
        with WorkingDirectoryChanger("../../../") as wd:
            sample = self.train_set[idx]
        return {
            "mixture": self.preprocessItem(sample[0]),
            "target": sample[1]
        }

    def __len__(self):
        return len(self.train_set)
