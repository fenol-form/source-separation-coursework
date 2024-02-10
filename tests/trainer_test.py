import logging

from torch.utils.data import DataLoader, Subset

from src.datasets.LibriMix import LibriMixDataset
from src.trainers.trainers import SeparationTrainer
from src.reporters.base import Reporter
from src.reporters.reporters import SeparationReporter
from src.models.models import DullModel

import hydra

from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.1", config_path=".", config_name="config_trainer_test")
def main(cfg: DictConfig):
    logger = logging.getLogger("trainer_test")

    model = hydra.utils.instantiate(
        cfg["model"],
        cfg["model"]["n_src"],
    )

    dataset = LibriMixDataset(datasetDir=cfg["dataset"]["dir"], mode="train", logger=logger)

    dataloader = DataLoader(dataset, batch_size=4)
    reporter = SeparationReporter(cfg, logger)
    metrics = {
        el["name"]: hydra.utils.instantiate(el["instance"])
        for el in cfg["metrics"]
    }

    lossModule = hydra.utils.instantiate(cfg["loss"])
    lossModule = lambda x, y: -lossModule(x, y)

    trainer = SeparationTrainer(
        model, metrics, metrics, cfg, lossModule, reporter, logger
    )

    trainer.train(dataloader)
    trainer.run(dataloader, dataloader, cfg["trainer"]["epochs"])


if __name__ == "__main__":
    main()
