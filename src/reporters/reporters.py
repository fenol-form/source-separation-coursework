from src.reporters.base import Reporter
import wandb


class SeparationReporter(Reporter):
    def __init__(self, config, logger):
        super().__init__(config, logger)

    def addAndReport(self, mode: str = "train", logs: dict = None, metricResults: dict = None):
        """
        Adds data to the buffer and reports when loggingPeriod ends and makes reset
        """
        self.mode = mode

        assert mode in ["train", "eval"], "mode should be either 'train' or 'eval"
        if mode == "train":
            try:
                loss = logs["loss"]
                gradNorm = logs["gradNorm"]
                wandb.log(self.wandbFormatNumber("loss", loss))
                wandb.log(self.wandbFormatNumber("gradNorm", gradNorm))
            except KeyError:
                self.logger.error("ERROR: some key is not in 'logs'")
                raise KeyError
        else:
            try:
                assert metricResults is not None, "metricResults should not be 'None' for 'eval' mode"
                for metricName, value in metricResults.items():
                    wandb.log(self.wandbFormatNumber(metricName, value))
            except KeyError:
                self.logger.error("ERROR: some key is not in 'logs'")
                raise KeyError

    def forceReport(self, mode="train"):
        pass

    def reset(self):
        pass
