import logging
import signal
import sys

import torch
from torch.utils.data import DataLoader, Subset

from src.datasets.LibriMix import LibriMixDataset
from src.inferencers.base import Inferencer
from src.trainers.trainers import SeparationTrainer
from src.reporters.base import Reporter
from src.reporters.reporters import SeparationReporter
from src.models.models import DPRNN

import hydra

from omegaconf import DictConfig, OmegaConf

from torch.profiler import profile, record_function, ProfilerActivity


@hydra.main(version_base="1.1", config_path=".", config_name="config_test_dprnn")
def main(cfg: DictConfig):
    logger = logging.getLogger("dprnn_test")

    model = hydra.utils.instantiate(
        cfg["model"],
        # cfg["model"]["n_src"],
    )

    dataset = LibriMixDataset(datasetDir=cfg["dataset"]["dir"], mode="train", logger=logger)
    dataset = Subset(dataset, [1, 2, 3])

    dataloader = DataLoader(dataset, batch_size=1)
    reporter = SeparationReporter(cfg, logger)
    metrics = {
        el["name"]: hydra.utils.instantiate(el["instance"])
        for el in cfg["metrics"]
    }

    lossModule = hydra.utils.instantiate(cfg["loss"])

    trainer = SeparationTrainer(
        model, metrics, metrics, cfg, lossModule, reporter, logger
    )
    inferencer = Inferencer(
        cfg, model, dataloader, reporter, metrics, logger
    )

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        with torch.no_grad():
            inferencer.validationRun(updatePeriod=50)
            # trainer.train(dataloader)
            # trainer.run(dataloader, dataloader, cfg["trainer"]["epochs"])
    print(prof.key_averages())


if __name__ == "__main__":
    main()
