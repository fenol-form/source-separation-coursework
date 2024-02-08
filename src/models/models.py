import torch
from torch import nn
from src.models.base import BaseModel


class DullModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.conv = nn.Conv1d(
            1,
            self.n_src,
            kernel_size=(3,),
            padding=1,
        )

    def forward(self, x):
        x = self.conv(x)
        return {"preds": x}

    def separate(self, mixture: torch.Tensor):
        with torch.no_grad():
            return self(mixture)
