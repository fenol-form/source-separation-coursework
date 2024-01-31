import logging

from torch.utils.data import DataLoader

from src.models.base import WavDPRNN
from src.datasets.LibriMix import LibriMixDataset
from src.inferencers.base import Inferencer
from src.reporters.base import Reporter
from src.metrics.metrics import *

import hydra

from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.1", config_path=".", config_name="config")
def main(cfg: DictConfig):
    logger = logging.getLogger("test")

    model = hydra.utils.instantiate(
        cfg["model"],
        n_src=cfg["model"]["n_src"],
        from_pretrained=cfg["model"]["from_pretrained"]
    )

    dataset = LibriMixDataset(datasetDir=cfg["dataset"]["dir"], mode="val", logger=logger)
    dataloader = DataLoader(dataset, batch_size=2)
    reporter = Reporter(cfg, logger)
    metrics = {
        el["name"]: hydra.utils.instantiate(el["instance"])
        for el in cfg["metrics"]
    }
    inferencer = Inferencer(
        cfg, model, dataloader, reporter, metrics, logger
    )

    inferencer.validationRun()


if __name__ == "__main__":
    main()
