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

    def separate(self, mixture: torch.Tensor):
        raise NotImplemented


class WavDPRNN(BaseModel):
    """
    Asteroid DPRNN wrapper
    """

    def __init__(self, *args, from_pretrained=False, **kwargs):
        super().__init__(*args)
        if from_pretrained:
            self.__asteroid_model = asteroid.models.BaseModel.from_pretrained(
                "mpariente/DPRNNTasNet-ks2_WHAM_sepclean"
            )
        else:
            self.__asteroid_model = asteroid.models.DPRNNTasNet(*args)

    def forward(self, x):
        return {"preds": self.__asteroid_model(x)}

    def separate(self, mixture: torch.Tensor) -> Tuple[torch.Tensor]:
        return self.__asteroid_model.separate(mixture)


class BaseMaskingNetwork(BaseModel):
    """
    Similar to https://github.com/asteroid-team/asteroid/blob/master/asteroid/models/base_models.py#L184
    """

    def __init__(
        self,
        n_src: int,
        encoder: nn.Module,
        decoder: nn.Module,
        masker: nn.Module,
        encoder_activation: nn.Module = None,
    ):
        super().__init__(n_src)
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder
        self.encoder_activation = encoder_activation

    def forward(self, x):
        """
        :param x: Tensor of shape (batch, in_chan, time frames)
        :return: Tensor of shape (batch, n_src, time frames)
        """
        tf_repr = self.encoder(x)

        if self.encoder_activation is not None:
            tf_repr = self.encoder_activation(tf_repr)

        est_masks = self.masker(tf_repr)
        masked_tf_repr = est_masks * tf_repr
        decoded = self.decoder(masked_tf_repr)

        assert decoded.shape[-1] == x.shape[-1]

        return decoded
