import logging
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.datasets.LibriMix import LibriMixDataset, DullDataset
from src.inferencers.base import Inferencer
from src.trainers.trainers import SeparationTrainer
from src.reporters.base import Reporter
from src.reporters.reporters import SeparationReporter
from src.models.models import MaskerDPRNN

import hydra
import torch

from omegaconf import DictConfig, OmegaConf

from torch.profiler import profile, record_function, ProfilerActivity


@hydra.main(version_base="1.1", config_path=".", config_name="config_test_dprnn")
def main(cfg: DictConfig):
    logger = logging.getLogger("dprnn_test")

    model = hydra.utils.instantiate(
        cfg["model"],
    )

    # dataset = LibriMixDataset(datasetDir=cfg["dataset"]["dir"], mode="train", logger=logger)
    dataset = DullDataset(datasetDir=cfg["dataset"]["dir"], mode="train", logger=logger)

    dataloader = DataLoader(dataset, batch_size=cfg["dataloader"]["batch_size"], shuffle=False)
    reporter = SeparationReporter(cfg, logger, sample_rate=8000)
    metrics = {
        el["name"]: hydra.utils.instantiate(el["instance"])
        for el in cfg["metrics"]
    }

    lossModule = hydra.utils.instantiate(cfg["loss"], func_type=cfg["loss"]["func_type"])

    trainer = SeparationTrainer(
        model, metrics, metrics, cfg, lossModule, reporter, logger
    )

    trainer.run(dataloader, dataloader, cfg["trainer"]["epochs"])


if __name__ == "__main__":
    main()