import asteroid
import torch
from torch import nn
from typing import Tuple


class BaseModel(nn.Module):

    def __init__(self, n_src):
        super().__init__()
        self.n_src = n_src

    def forward(self, x):
        raise NotImplemented

    def separate(self, mixture: torch.Tensor) -> Tuple[torch.Tensor]:
        raise NotImplemented


class WavDPRNN(BaseModel):
    """
    Asteroid DPRNN wrapper
    """

    def __init__(self, *args, from_pretrained=True, **kwargs):
        super().__init__(*args, **kwargs)
        if from_pretrained:
            self.__asteroid_model = asteroid.models.BaseModel.from_pretrained(
                "mpariente/DPRNNTasNet-ks2_WHAM_sepclean"
            )
        else:
            self.__asteroid_model = asteroid.models.DPRNNTasNet(*args, **kwargs)

    def forward(self, x):
        return self.__asteroid_model(x)

    def separate(self, mixture: torch.Tensor) -> Tuple[torch.Tensor]:
        return self.__asteroid_model.separate(mixture)
