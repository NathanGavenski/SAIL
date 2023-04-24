from collections import defaultdict
import os
import shutil

import numpy
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


class Board():
    def __init__(self, name=None, path=None):
        path = './runs/' if path is None else path
        if os.path.exists(path) is False:
            os.makedirs(path)

        if os.path.exists(f'{path}') is True:
            shutil.rmtree(f'{path}')

        if name is None:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter(f'{path}')

        self.epoch = 0
        self.histograms = defaultdict(list)

    def advance(self):
        if self.histograms:
            self.__save_histogram()
            self.histograms = defaultdict(list)
        self.epoch += 1

    def close(self):
        self.writer.flush()
        self.writer.close()

    def add_grid(self, train, **kwargs):
        for i in kwargs:
            grid = torchvision.utils.make_grid(kwargs[i])
            if train:
                self.writer.add_image(f'Train/{i}', grid, self.epoch)
            else:
                self.writer.add_image(f'Validation/{i}', grid, self.epoch)

    def add_image(self, title, value):
        self.writer.add_image(title, value, self.epoch)

    def add_scalars(self, train, **kwargs):
        for i in kwargs:
            if train:
                self.writer.add_scalar(f'Train/{i}', kwargs[i], self.epoch)
            else:
                self.writer.add_scalar(f'Validation/{i}', kwargs[i], self.epoch)

    def add_histogram(self, title, histogram):
        if isinstance(histogram, torch.Tensor) and histogram.is_cuda:
            histogram = torch.flatten(histogram).cpu().tolist()
        elif isinstance(histogram, torch.Tensor):
            histogram = torch.flatten(histogram).tolist()

        self.histograms[title].extend(histogram)

    def save_histogram(self, name):
        titles = []
        for title in self.histograms:
            if name in title:
                self.writer.add_histogram(title, numpy.array(self.histograms[title]), self.epoch)
                titles.append(title)

        for title in titles:
            self.histograms.pop(title, None)

    def add_hparams(self, params):
        self.writer.add_hparams(params)

    def __save_histogram(self):
        for title in self.histograms:
            self.writer.add_histogram(title, numpy.array(self.histograms[title]), self.epoch)

    def add_scalar(self, title, value):
        self.writer.add_scalar(title, value, self.epoch)
