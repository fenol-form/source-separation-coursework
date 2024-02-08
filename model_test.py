from asteroid.models import BaseModel
import soundfile as sf
from src.datasets.LibriMix import LibriMixDataset
from src.models.base import WavDPRNN
import logging
import hydra

from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.1", config_path="tests", config_name="config")
def main(cfg: DictConfig):
    logger = logging.getLogger("test")

    model = hydra.utils.instantiate(
        cfg["model"],
        n_src=cfg["model"]["n_src"],
        from_pretrained=cfg["model"]["from_pretrained"]
    )

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    dataset = LibriMixDataset(datasetDir=cfg["dataset"]["dir"], mode="val", logger=logger)
    mixture = dataset[0][0].unsqueeze(0).unsqueeze(0)

    out_wavs = model(mixture)
    print(out_wavs.shape)


main()


