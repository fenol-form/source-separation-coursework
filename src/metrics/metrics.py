from typing import Literal

import torch
from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
from torchmetrics.functional.audio import signal_noise_ratio, signal_distortion_ratio
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torch import nn


class BasePITMetric(nn.Module):
    """
    Base class for all metric which use PIT
    """

    def __init__(self, func=None, eval_func=None, **kwargs):
        super().__init__()
        self.pit_loss = PermutationInvariantTraining(metric_func=func, eval_func=eval_func, **kwargs)

    def forward(self, preds, target):
        assert preds.shape == target.shape
        if len(preds.shape) <= 1:
            preds = preds.unsqueeze(0).unsqueeze(0)
        if len(target.shape) <= 1:
            target = target.unsqueeze(0).unsqueeze(0)
        return self.pit_loss(preds, target)


class SNRMetric(BasePITMetric):

    def __init__(self, func_type: Literal["snr", "si-snr"] = "snr"):
        if func_type == "snr":
            super().__init__(func=signal_noise_ratio, eval_func="max")
        elif func_type == "si-snr":
            super().__init__(func=scale_invariant_signal_noise_ratio, eval_func="max")
        else:
            raise ValueError(f"func_type has to be in [snr, si-snr]")


class SDRMetric(BasePITMetric):

    def __init__(self, func_type: Literal["sdr", "si-sdr"] = "sdr"):
        if func_type == "sdr":
            super().__init__(func=signal_distortion_ratio, eval_func="max")
        elif func_type == "si-sdr":
            super().__init__(func=scale_invariant_signal_distortion_ratio, eval_func="max")
        else:
            raise ValueError(f"func_type has to be in [sdr, si-sdr]")


class PESQMetric(BasePITMetric):

    def __init__(self, fs, mode):
        super().__init__(
            func=perceptual_evaluation_speech_quality,
            eval_func="max",
            fs=fs, mode=mode
        )


class NegSNR(SNRMetric):
    def forward(self, preds, target):
        return -super().forward(preds, target)


class NegSDR(SDRMetric):
    def forward(self, preds, target):
        return -super().forward(preds, target)

