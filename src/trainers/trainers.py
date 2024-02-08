from src.trainers.base import *


class SeparationTrainer(Trainer):
    def __init__(self, model, metrics, finalMetrics, config, lossModule, reporter, logger):
        super().__init__(model, metrics, finalMetrics, config, lossModule, reporter, logger)

    def computeLoss(self, batch):
        """
        Computes loss function to return to backprop + some tech data

        ------------------------
        Parameters
        dict batch -- batch in the form of dict
        """
        x = batch["mixture"]
        batch.update(self.model(x))
        loss = self.lossModule(batch["preds"], batch["target"])
        batch["loss"] = loss

        return batch

    def extractPrediction(self, batch):
        if "preds" in batch:
            return batch["preds"]
        else:
            self.logger.error("ERROR: Preds are not in batch!")
            raise ValueError

    def extractTarget(self, batch):
        if "target" in batch:
            return batch["target"]
        else:
            self.logger.error("ERROR: Target is not in batch!")
            raise ValueError
