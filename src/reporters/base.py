import numpy as np

import torch
import torchaudio

import pandas as pd

import os

import wandb
import soundfile as sf


class Reporter:

    def __init__(self, config, logger):
        """
        Reporter base class
        logger.Logger logger -- logger object to use
        """
        self.logger = logger
        self.loggingPeriod = config["logs"]["loggingPeriod"]
        try:
            self.wandbCredentials = config["wandbCredentials"]
        except:
            self.wandbCredentials = None
            self.logger.info("WARNING: Reporter could not read wandbCredentials, wandb logs are turned off")
        # init wandb
        if not (self.wandbCredentials is None):
            os.environ["WANDB_API_KEY"] = self.wandbCredentials["wandbKey"]  # YOUR API KEY HERE
            wandb.init(
                project=self.wandbCredentials["wandbProject"],
                entity=self.wandbCredentials["wandbEntity"],  # YOUR USERNAME
                name=self.wandbCredentials["runName"],
                config=dict(config)
            )

    def addAndReport(self, mode="train", report=False, **reportData):
        """
        Adds the data to buffer and reports when loggingPeriod ends and makes reset()
        """
        self.logger.info("WARNING: method 'addAndReport' is not implemented yet!")
        # raise NotImplementedError

    def forceReport(self, mode="train"):
        """
        Sends the remaining logs and flushes the buffer and makes reset()
        """
        self.logger.info("forceReport")
        # raise NotImplementedError

    def reset(self):
        """
        Resets the buffer
        """
        self.logger.info("Reporter.reset()")
        # raise NotImplementedError

    def wandbFormatName(self, name):
        return f"{name}_{self.mode}"

    def wandbFormatNumber(self, numberName, number):
        return {
            self.wandbFormatName(numberName): number
        }

    def wandbFormatNumbers(self, tag, numbers):
        return {
            **{f"{numberName}_{tag}_{self.mode}": number for numberName, number in
               numbers.items()}
        }

    def wandbFormatHistogram(self, histName, xs, numBins=64):
        return {
            histName: wandb.Histogram(xs, num_bins=numBins)
        }

    def wandbFormatImage(self, imageName, image):
        return {
            self.wandbFormatName(imageName): wandb.Image(image)
        }

    def wandbFormatAudio(self, audioName, audio, sampleRate=None):
        audio = audio.detach().cpu().numpy().T
        return {
            self.wandbFormatName(audioName): wandb.Audio(audio, sample_rate=sampleRate)
        }

    def wandbFormatAudios(self, audios, sampleRate):
        audios = {
            f"{i}-th speaker": audios[i] for i in range(audios.size(0))
        }
        return {
            audioName: wandb.Audio(audio, sample_rate=sampleRate) for audioName, audio in
            audios.items()
        }

    def wandbFormatTable(self, tableName, columns):
        return {
            tableName: wandb.Table(columns=columns)
        }

    def wandbFormatText(self, textName, text):
        return {
            self.wandbFormatName(textName): wandb.Html(text)
        }
