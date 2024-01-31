import logging

import numpy as np
import torch
import torchaudio

import soundfile as sf

from torch.utils.data import Dataset

import hydra
from omegaconf import OmegaConf


# For now BaseDataset is just a table of all audio files
class BaseDataset(Dataset):
    def __init__(self, mode="train", datasetDir=None, preprocessing=None,  index=None, logger=None):
        self.datasetDir = datasetDir
        self.logger = logger  # logger to use
        self.index = index  # list of dicts with metadata for each audio
        self.preprocessing = preprocessing  # config preprocessing section
        self.mode = mode
        try:
            if preprocessing["modules"] is None:
                self.preprocessingModules = []
            else:
                self.preprocessingModules = [hydra.utils.instantiate(self.preprocessing["modules"][i]) for i in
                                             range(len(self.preprocessing["modules"]))]
        except Exception as e:
            self.logger.info("WARNING BaseDatasetInit: something wrong with preprocessing modules, setting []")
            self.logger.info(str(e))
            self.preprocessingModules = []

    def __getitem__(self, key):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def loadAudio(self, path):
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
        with torch.no_grad():
            for i in np.arange(len(self.preprocessingModules)):
                audioTensor = self.preprocessingModules[i](audioTensor)
            return audioTensor
#
#
# class VADProcessor:
#     '''
#     Under testing, breaks gradient somehow => not recommended to use on training
#     '''
#
#     def __init__(self, sr=16000, model_name='silero_vad', minSilence=1000):
#         self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model=model_name)
#         self.model.eval()
#         self.sr = sr
#         # get_speech_timestamps - это функция, которая принимает аудиофайл и модель VAD и возвращает список словарей, содержащих начало и конец речи в секундах
#         # save_audio - это функция, которая сохраняет аудиофайл на
#         # read_audio - это функция, которая читает аудиофайл и возвращает numpy-массив с данными
#         # collect_chunks - это функция, которая собирает чанки аудио в один файл
#         self.minSilence = minSilence
#         (self.get_speech_timestamps,
#          self.save_audio,
#          self.read_audio,
#          self.VADIterator,
#          self.collect_chunks) = self.utils
#
#     def __call__(self, audio):
#         # self.model.to(audio.get_device())
#
#         timestamps = self.get_speech_timestamps(audio.to(dtype=torch.float32), self.model, sampling_rate=self.sr,
#                                                 min_silence_duration_ms=self.minSilence)
#         speech = self.collect_chunks(timestamps, audio.to(dtype=torch.float32))
#         speech.requires_grad = False
#         print("SPEECH", speech.shape, speech)
#         return speech

