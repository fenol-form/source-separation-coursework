import torch
import numpy as np

from collections import deque

import os
import sys

import hydra

from src.reporters.base import Reporter
from src.utils.utils import loadObj


class TrainerError(Exception):
    def __init__(self):
        self.msg = "TrainerError"

    def __str__(self):
        return self.msg


class NewCheckpointsPathIsNotSetException(TrainerError):
    def __init__(self):
        super(NewCheckpointsPathIsNotSetException, self).__init__()

    def __str__(self):
        return self.msg + ": Path for new checkpoints is not set, check your config"


class Trainer(object):
    """
    Trainer base class: object used to run training and eval routines
    """

    def __init__(self, model, metrics, finalMetrics, config, lossModule, reporter, logger):
        """
        Input
        nn.Module model -- taken from src.models
        YAML config -- config
        """
        self.config = config

        # logging
        self.logger = logger
        self.reporter = reporter

        # choosing the device
        if torch.cuda.is_available():
            self.logger.info("CUDA is available, using GPU for computations")
        else:
            self.logger.info("CUDA is unavailable, using CPU for computations")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # metrics and loss
        self.metrics = loadObj(metrics, self.device)
        self.finalMetrics = loadObj(finalMetrics, self.device)
        self.lossModule = loadObj(lossModule, self.device)

        # model
        self.model = model.to(self.device)  # send the model to devic

        # training init stuff
        self.trainableParams = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = hydra.utils.instantiate(self.config["optimizer"], params=self.trainableParams)
        try:
            self.lrScheduler = hydra.utils.instantiate(self.config["lrScheduler"], optimizer=self.optimizer)
        except KeyError:
            self.logger.info("WARNING: LR scheduler is not set, proceeding with standard one")

        # init epochCounter
        self.curEpoch = 0

        # checkpoint handling
        self.checkpointPath = config.checkpointPath
        if self.checkpointPath is not None:
            self.logger.info(f"Continue training from checkpoint: {self.checkpointPath}")
            checkpoint = torch.load(self.checkpointPath, map_location='cpu')
            try:
                self.model.load_state_dict(checkpoint['model'])
            except Exception as e:
                self.logger.info(e)
                self.logger.info("WARNING! load_state_dict_failed, expected load_state_dict in the child init")
            self.model = self.model.to(self.device)
        else:
            self.logger.info("Starting new training run")

            # checkpoint configuration
        self.checkpointQueue = deque(maxlen=config.nCheckpoints)
        self.newCheckpointsPath = config["newCheckpointsPath"]

        if self.newCheckpointsPath and not os.path.exists(self.newCheckpointsPath):
            self.logger.info("WARNING: newCheckpointsPath does not exist, creating one at " + self.newCheckpointsPath)
            os.mkdir(self.newCheckpointsPath)
        else:
            raise NewCheckpointsPathIsNotSetException()

    def processCheckpoint(self, path):
        if len(self.checkpointQueue) == self.checkpointQueue.maxlen:
            removedCheckpoint = self.checkpointQueue[0]
            os.remove(removedCheckpoint)
        self.checkpointQueue.append(path)

    def checkBestCheckpoint(self):
        return NotImplementedError()

    def saveCheckpoint(self, idx):
        """
        Saves the checkpoint
        Input
        int idx -- checkpointId (usually epoch number)
        """
        cpt = {
            "epoch": self.curEpoch,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lrScheduler.state_dict(),
            "model": self.model.state_dict()
        }
        pathToSave = os.path.join(self.newCheckpointsPath,
                                  "{0}.pt".format(str(idx)))
        torch.save(cpt, pathToSave)

        self.processCheckpoint(pathToSave)

    def computeLoss(self, egs):
        """
        Computes loss based on the observed batch with predictions.
        Input
        DataLoaderChunk egs -- dataloader item

        Output
        torch.Tensor loss, Dict batch
                    loss  must be differentiable to do backward
        """
        raise NotImplementedError

    def computeMetrics(self, preds, targets):
        """
        Computes metrics based on the comparison of predictions and targets

        Parameters
        ------------------------
        torch.Tensor preds -- predictions (N,)
        torch.Tensor targets -- targets (N,)
        """
        return {key: metric(preds, targets) for key, metric in self.metrics.items()}

    def computeFinalMetrics(self, metricsValues, dataLoader):
        """
        Computes final metric values by averaging it

        Parameters
        ------------------------
        torch.Tensor metricValues -- dict[List]
        torch.Tensor dataLoader -- torch.utils.data.DataLoader
        """
        dataloader_len = len(dataLoader)
        return {
            key: sum(values) / dataloader_len for key, values in metricsValues.items()
        }

    def calcGradNorm(self):
        """
        Computes square norm of the gradient to check stability
        """
        with torch.no_grad():
            totalNorm = 0.0
            parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters:
                paramNorm = torch.mean(p.grad.detach().data ** 2, 0).sum()
                totalNorm += paramNorm
            totalNorm = 0.5 * torch.log(totalNorm)
        return totalNorm

    def train(self, dataLoader):
        self.logger.info("Set train mode...")
        self.model.train()
        mode = "train"
        numSteps = len(dataLoader)

        for step, batch in enumerate(dataLoader):
            batch = loadObj(batch, self.device)  # load to self.device

            # do gradient stuff
            self.optimizer.zero_grad()
            batch = self.computeLoss(batch)
            loss = batch["loss"]
            self.logger.info("LOSS " + str(loss))
            loss.backward()
            # update optimizer and scheduler (if any)
            self.optimizer.step()
            if self.lrScheduler is not None:
                self.lrScheduler.step()
            # add tech logs
            logs = {}
            logs.update({"progress": (step + 1 + (numSteps * self.curEpoch)) / numSteps})
            logs.update({"loss": batch["loss"].detach().cpu().numpy()})
            logs.update({"step": (step + 1 + (numSteps * self.curEpoch))})
            logs.update({"gradNorm": self.calcGradNorm().cpu().numpy()})
            # report logs
            self.reporter.addAndReport(logs=logs, mode=mode)

        self.reporter.forceReport(mode)  # to free buffer, flush the remaining logs

    def extractPrediction(self, batch):
        """
        Extracts predictions from the batch (model-dependent format)

        ------------------------
        Parameters
        dict batch -- data element

        Returns
        float[] preds -- tensor of predictions
        """
        raise NotImplementedError

    def extractTarget(self, batch):
        """
        Extracts the target of the model from dataElement

        Parameters
        -----------------
        dict batch -- dataset element, formally dict

        Returns
        float[] targets -- tensor of targets
        """
        raise NotImplementedError

    def eval(self, dataLoader):
        self.logger.info("Set eval mode...")
        self.model.eval()
        mode = "eval"

        outData = []
        metrics = {metricName: [] for metricName in self.metrics.keys()}
        dataset_len = len(dataLoader)

        with torch.no_grad():
            for batch in dataLoader:
                batch = loadObj(batch, self.device)  # send the batch to device
                batch = self.computeLoss(batch)  # compute loss and logs
                pred = self.extractPrediction(batch)
                targ = self.extractTarget(batch)

                # update running metrics
                batch_metrics = self.computeMetrics(pred, targ)
                for metricName in self.metrics:
                    metrics[metricName].append(batch_metrics[metricName])

                # making loss data 1d
                batch["loss"] = torch.mean(batch["loss"])

                outData.append(batch)

            # self.logger.info("PREDS " + str(preds))
            finalMetricsResult = self.computeFinalMetrics(metrics, dataLoader)

            # TODO: change...
            metricsResult = finalMetricsResult

            self.reporter.step = self.curEpoch
            self.reporter.addAndReport(curEpoch=self.curEpoch, outData=outData, metricsResult=metricsResult,
                                       finalMetricsResult=finalMetricsResult, mode=mode)  # report logs

        self.reporter.forceReport(mode)  # to free buffer, flush the remaining logs

    def run(self, trainLoader, testLoader, nEpochs=50):
        self.saveCheckpoint(self.curEpoch)  # anyway, just save the first one
        self.reporter.step = 0  # set up attribute
        self.eval(testLoader)
        while self.curEpoch < nEpochs:
            self.logger.info("Initiating epoch " + str(self.curEpoch))
            self.curEpoch += 1

            self.train(trainLoader)
            self.eval(testLoader)

            self.saveCheckpoint(self.curEpoch)
            sys.stdout.flush()  # just in case, flush the logs in the logger

        self.logger.info(f"Training for {self.curEpoch}/{nEpochs} epoches done!")