import os
import sys

import hydra
import torch
import torchaudio
import soundfile as sf

import numpy as np
import pandas as pd

import time

from src.utils.utils import loadObj

sys.path.append("../")


class InferenceError(Exception):
    # Base class for exceptions
    def __init__(self):
        self.msg = "Base Inference Exception, something is wrong..."

    def __str__(self):
        # print it
        return self.msg


class AudioTypeError(InferenceError):
    def __init__(self, audioType):
        self.msg = f"No such audioType: {audioType}, you can use only 'set' or 'topic' "


class Inferencer(object):
    """
    Inferencer base class: object used to run simple and validation inference
    """

    def __init__(self, config, model, dataloader, reporter, metrics, logger):
        """
        Parameters
        -------------------------------
        LoadConfig config -- config
        TorchModel model -- taken from src.models
        src.datasets.base.BaseDataset dataset -- dataset instance
        Reporter reporter -- Reporter object
        Metric[] metrics -- dict of metrics to apply
        logger -- logger object
        """

        self.config = config

        # logging
        self.logger = logger
        self.reporter = reporter

        # metrics
        self.metrics = metrics

        # dataset
        self.dataloader = dataloader  # setting up the dataset

        # output
        self.outPath = config["inference"]["outPath"]
        self.outPathMetrics = config["inference"]["outMetricsPath"]

        # load the model
        self.model = model
        self.loadModel()

    def loadModel(self):
        """
        Makes all the necessary routines to load model checkpoint and misc stuff
        (Here assuming torch-like model but can be overrided in the child classes)
        """
        self.preprocessing = self.config["inference"][
            "preprocessing"]  # a separate preprocessing for just one separate inference run
        try:
            if self.preprocessing["modules"] is None:
                self.preprocessingModules = []
            else:
                self.preprocessingModules = [hydra.utils.instantiate(self.preprocessing["modules"][i]["args"]) for i in
                                             range(len(self.preprocessing["modules"]))]
        except Exception as e:
            self.logger.info("WARNING loadModel: something wrong with preprocessing modules, setting []")
            self.logger.info(str(e))
            self.preprocessingModules = []

        # choose the device
        if torch.cuda.is_available():
            self.logger.info("CUDA is available, using GPU for computations")
        else:
            self.logger.info("CUDA is unavailable, using CPU for computations")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)  # send the model there

        self.checkpointPath = self.config["checkpointPath"]
        self.logger.info("Using checkpoint: %s" % self.checkpointPath)

        # set up the checkpoint
        if self.checkpointPath is None:
            self.logger.info("WARNING! checkpointPath is None, are you using BaseModelInferencer?")
        else:
            checkpoint = torch.load(self.checkpointPath, map_location='cpu')

        try:
            self.model.load_state_dict(checkpoint['model'])
        except:
            self.logger.info("WARNING! load_state_dict_failed, expected load_state_dict in the child init")

        self.model.eval()

    def oneRun(self, datasetEl):
        """
        Just runs the models and returns the result of the evaluation in the setting of validationRun

        Parameters
        ---------------
        dict datasetEl -- dataset element
        or
        float[] audio
        """
        with torch.no_grad():
            if not (self.device == "cpu"):
                datasetEl = loadObj(datasetEl, self.device)
            return self.model(datasetEl)

    def extractBatch(self, dataEl):
        if "target" in dataEl:
            return dataEl["mixture"], dataEl["target"]
        return dataEl["mixture"]

    def extractPrediction(self, model_out):
        """
        Extracts the prediction of the model from dataElement

        Parameters
        -----------------
        DatasetElement dataEl -- dataset element, formally dict
        """
        return model_out["preds"]

    def extractTarget(self, dataEl):
        """
        Extracts the target of the model from dataElement

        Parameters
        -----------------
        DatasetElement dataEl -- dataset element, formally dict
        """
        return NotImplementedError

    def computeMetrics(self, preds, targets):
        """
        Computes metrics based on the comparison of predictions and targets

        Parameters
        ------------------------
        torch.Tensor preds -- predictions (N,)
        torch.Tensor targets -- targets (N,)
        """
        resultDict = {metricName: sum([self.metrics[metricName](pred, target).item()
                                       for pred, target in zip(preds, targets)]) / len(preds)
                      for metricName in self.metrics.keys()}
        return [{metricName: resultDict[metricName]} for metricName in resultDict.keys()]

    def processPreds(self, preds) -> torch.Tensor:
        return preds

    def processTargets(self, targets):
        """
        Transforms targets in format [{ grade1X: , grade1Y: .....}]
        to format {grade1X: ..., grade1Y: ... }
        """
        return targets

    def validationRun(self, updatePeriod=50):
        """
        Runs a session of validation inference

        Parameters
        --------------------------------
        str outPath -- path to the log dump
        int updatePeriod -- ??
        """

        curIter = 0
        time0Overall = time.time()

        outData = []
        preds = []
        targets = []
        with torch.no_grad():
            for el in self.dataloader:
                try:
                    time0Inference = time.time()  # inference time measurement
                    el, target = self.extractBatch(el)
                    out = self.oneRun(el)
                    t0 = time.time() - time0Inference
                    preds.append(self.extractPrediction(out))
                    targets.append(target)

                    out.update({"inferenceTime": t0})

                    del out["preds"]
                    outData.append(out)
                except ValueError as e:
                    self.logger.info(str(e))
                    self.logger.info(f"Error on element: {el}")

                if curIter % updatePeriod == 0:
                    durationOverall = time.time() - time0Overall
                    self.logger.info(
                        f"Files Processed | {curIter + 1} out of {len(self.dataloader)}; {((curIter + 1) / len(self.dataloader)) * 100:.2f}% in time {durationOverall}")
                    time0Overall = time.time()
                curIter += 1

        preds = self.processPreds(preds)
        targets = self.processTargets(targets)

        metrics = self.computeMetrics(preds, targets)
        self.reporter.forceReport()
        if self.outPath:
            self.saveInferencerData(outData, metrics)

        self.reporter.reset()

    def saveInferencerData(self, outData, metrics):
        """
        Saves inferencer data into file

        Parameters
        -----------
        List outData -- list of observations (with removed audio)
        dict metrics -- metrics data
        str outPath -- path to file to save
        """
        df = pd.DataFrame(outData)
        df.to_csv(self.outPath)
        df2 = pd.DataFrame(metrics)
        df2.to_csv(self.outPathMetrics)

    def loadAudio(self, path):
        """
        Loads audio located at path. Notice: depends on soundfile, audio is preferably in mp3 or wav or OGG(Vorbis only)
        Input
        str path -- path to the audio file
        """
        audioTensor, sr = sf.read(path)
        # force mono audio
        # anyway, it's mono (N,)....but sometimes not
        if len(audioTensor.shape) > 1:
            # most probably it's the smallest dim unless you have 2-sample audio
            minDim = np.where(np.array(audioTensor).shape == np.amin(np.array(audioTensor.shape)))[0][0]
            # print("MinDim", minDim)
            audioTensor = np.mean(audioTensor, axis=minDim)
        audioTensor = torch.from_numpy(audioTensor)
        srToUse = self.preprocessing["sr"]
        if not (sr == srToUse):
            audioTensor = torchaudio.functional.resample(audioTensor, sr, srToUse)
        return audioTensor

    def preprocessAudio(self, audioTensor):
        """
        Preprocesses audio before sending to the model
        """
        with torch.no_grad():
            for i in np.arange(len(self.preprocessingModules)):
                audioTensor = self.preprocessingModules[i](audioTensor)
            return audioTensor

    def oneAudioRun(self, audioPath):
        """
        Preprocesses audio before sending to the model
        Input
        str audioPath -- path to audio file
        str audioType -- model to use (set of topic)
        """
        audio = self.preprocessAudio(self.loadAudio(audioPath))
        return self.oneRun(audio)
