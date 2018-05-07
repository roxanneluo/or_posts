
import json, logging, pickle, sys
import os
from os import path
from torchnlp.text_encoders import IdentityEncoder, WhitespaceEncoder
import torch
import torch.nn as nn

kPrepDataDir = 'prep'
kVocabStart = 5

class EncoderBase:
    def __init__(self, start = kVocabStart):
        self.start = start

    def vocab(self):
        return self.encoder.vocab[self.start:]

    def encode(self, labels):
        f = lambda x: x - self.start
        results = self.encoder.encode(labels)
        if isinstance(labels, str):
            return f(results)
        return list(map(f, results))

    def decode(self, indices):
        f = lambda x: x + self.start
        results = self.encoder.decode(indices)
        if isinstance(labels, str):
            return f(results)
        return list(map(f, results))

class LabelEncoder(EncoderBase):
    def __init__(self, data_path):
        super().__init__()

        save_path = path.join(kPrepDataDir, 'labels.pt')
        if path.exists(save_path):
            with open(save_path, 'rb') as f:
                labels = pickle.load(f)
            self.encoder = IdentityEncoder(labels)
            return

        labels = []
        with open(data_path, 'r') as f:
            for line in f:
                label = line.split('\t')[-1][:-1]
                labels.append(label)
        self.encoder = IdentityEncoder(labels)
        with open(save_path, 'wb') as f:
            pickle.dump(self.vocab(), f)

class Encoder(EncoderBase):
    def __init__(self, text = None):
        offset = 1
        super().__init__(kVocabStart - offset)

        save_path = path.join(kPrepDataDir, 'vocab.pt')
        if path.exists(save_path):
            with open(save_path, 'rb') as f:
                vocabulary = pickle.load(f)
            self.encoder = WhitespaceEncoder(vocabulary)
            return

        self.encoder = WhitespaceEncoder(text)
        with open(save_path, 'wb') as f:
            pickle.dump(self.vocab()[offset:], f)


