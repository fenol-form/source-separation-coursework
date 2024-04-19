import logging

import asteroid.models
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.datasets.LibriMix import LibriMixDataset, DullDataset
from src.datasets.OneSampleSet import OneSampleSet
from src.inferencers.base import Inferencer
from src.trainers.trainers import SeparationTrainer
from src.reporters.base import Reporter
from src.reporters.reporters import SeparationReporter
from src.models.models import MaskerDPRNN, DPRNN
import src.models.base
import src.models.models

import hydra
import torch

from omegaconf import DictConfig, OmegaConf

from torch.profiler import profile, record_function, ProfilerActivity


@hydra.main(version_base="1.1", config_path=".", config_name="config_dprnn_debug")
def main(cfg: DictConfig):
    logger = logging.getLogger("dprnn_test")

    model = hydra.utils.instantiate(
        cfg["model"],
    )

    # correct run name
    if isinstance(model, src.models.base.AsteroidDPRNN):
        cfg["wandbCredentials"]["runName"] += "__asteroid_model"
    elif isinstance(model, src.models.models.DPRNN):
        cfg["wandbCredentials"]["runName"] += "__my_model"
    else:
        cfg["wandbCredentials"]["runName"] += "__some_model"

    # use one sample dataset for debugging
    # dataset = DullDataset(datasetDir=cfg["dataset"]["dir"], mode="train", logger=logger)
    dataset = OneSampleSet(mode="train", logger=logger)

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
