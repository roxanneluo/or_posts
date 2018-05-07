
import json, logging, pickle, sys
import os
from os import path
from torchnlp.text_encoders import IdentityEncoder, WhitespaceEncoder
import torch
import torch.nn as nn

kPrepDataDir = 'prep'
kVocabStart = 5

class LabelEncoder:
    def __init__(self, data_path):
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
        print(self.vocab(), len(self.vocab()))
        with open(save_path, 'wb') as f:
            pickle.dump(self.vocab()[kVocabStart:], f)

    def vocab(self):
        return self.encoder.vocab[kVocabStart:]

    def encode(self, labels):
        f = lambda x: x-kVocabStart
        results = self.encoder.encode(labels)
        if isinstance(labels, str):
            return f(results)
        return list(map(f, results))

    def decode(self, indices):
        return self.encoder.decode(indices)

class Encoder:
    pass



"""

    train [[1,2,3], ...] ,label[]
    dev
    unlabeled [[], ....]

    train_combined = train + unlabeled


    read all texts => unlabeled [' dsfsd', 'sdfsd'] labeled, name_name_to_index dict for labeled
    "encoder" = build encoder from labeld + unlabeled
    "labeld_encoder" = label_encoder
    "labeled_x, labeled_y" = build labeled(labeled, labeled + name2idx dict, 'labeled.tsv', encoder, label_encoder)
    "dev_x", "dev_y" = build labeled(dev, dev+ name2idx dict, 'dev.tsv')

"""
