import torch
import numpy as np
import os
from os import path
from encoder import kPrepDataDir
import pickle
from pdb import set_trace as st
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

class DataLoader:
    def __init__(self,
            save_name,
            label_path = None,
            encoder = None, encoder_y = None,
            text = None, fn2idx=None,
            tensor=torch.LongTensor,
            batchSize=64, use_cuda=False,
            is_train=True, ratio=0.9,
            stopEnough=10000000):

        filename = path.join(kPrepDataDir, save_name)
        if os.path.exists(filename):
            print("Loading dataset from %s" % filename)
            with open(filename, 'rb') as f:
                state = pickle.load(f)
                self.x, self.y, self.fn2idx = state['x'], state['y'], state['fn2idx']
        else:
            self.x, self.y = self.construct_dataset(label_path, text, fn2idx, encoder, encoder_y)
            self.fn2idx = fn2idx
            with open(filename, 'wb') as f:
                state = {'x': self.x, 'y': self.y, 'fn2idx': self.fn2idx}
                state = pickle.dump(state, f)

    def construct_dataset(self, label_path, text, fn2idx, encoder, encoder_y):
        x, y = [],[]
        if label_path:
            with open(label_path, 'r') as f:
                for line in f:
                    fn, label = line[:-1].split('\t')
                    t = text[fn2idx[fn]]
                    t_encoded = encoder.encode(t)
                    y_encoded = encoder_y.encode(label)
                    x.append(t_encoded)
                    y.append(y_encoded)
            return x, y

        for t in text:
            x.append(encoder.encode(t))
            y.append(0)
        return x, y

    def data(self, num_categories = 19,  maxlen = None):
        x = pad_sequences(self.x, maxlen=maxlen, dtype='int32', padding='post', truncating='post', value = 0)
        y = to_categorical(self.y, num_classes = num_categories)
        return x, y

    def __len__(self):
        return len(self.y)

    def dump(self, f, encoder_y):
        for fn, idx in self.fn2idx.items():
            f.write(fn + '\t' + encoder_y.decode(self.y[idx]) + '\n')

def readFilesInDir(dir_path, build_dict = False):
    save_path = path.join(kPrepDataDir, path.basename(dir_path) + '.pickle')
    if path.exists(save_path):
        with open(save_path, 'rb') as f:
            state = pickle.load(f)
            text, fn2idx = state['text'], state['fn2idx']
        return text, fn2idx

    text = []
    fn2idx = {}
    for name in os.listdir(dir_path):
        if build_dict:
            fn2idx[path.join(path.basename(dir_path), name)] = len(text)
        with open(path.join(dir_path, name), 'r') as f:
            line = f.read()
        text.append(line[:-1])

    # dump
    state = {'text': text, 'fn2idx': fn2idx}
    with open(save_path, 'wb') as f:
        pickle.dump(state, f)

    return text, fn2idx


def loadCorpus(labeled_fn, unlabeled_fn):
    labeled, fn2idx = readFilesInDir(labeled_fn, build_dict = True)
    unlabeled, fn2idx_unlabeled = readFilesInDir(unlabeled_fn, build_dict = True)
    return unlabeled, fn2idx_unlabeled, labeled, fn2idx

if __name__ == '__main__':
    from encoder import Encoder, LabelEncoder

    os.makedirs('prep', exist_ok = True)

    data_dir = path.join('..', 'dataset')
    encoder_y = LabelEncoder(path.join(data_dir, 'train.tsv'))
    print('#labels=', len(encoder_y.vocab()), encoder_y.vocab())
    unlabeled_text, fn2idx_unlabeled, labeled_text, fn2idx = loadCorpus(path.join(data_dir, 'labeled'),
            path.join(data_dir, 'unlabeled'))
    print('#labeled=', len(labeled_text), '#unlabeled=', len(unlabeled_text))
    encoder = Encoder(labeled_text + unlabeled_text)
    print('#words=', len(encoder.vocab()), encoder.vocab()[:10])

    train_loader = DataLoader(label_path=path.join(data_dir, 'train.tsv'),
            text=labeled_text, fn2idx = fn2idx,
            encoder=encoder, encoder_y=encoder_y, save_name='train.pickle')
    print('#train', len(train_loader))
    train_x, train_y = train_loader.data()
    print('padded train shape', train_x.shape, train_y.shape)

    dev_loader = DataLoader(label_path=path.join(data_dir, 'dev.tsv'),
            text=labeled_text, fn2idx = fn2idx,
            encoder=encoder, encoder_y=encoder_y, save_name='dev.pickle')
    print('#dev', len(dev_loader))
    dev_x, dev_y = dev_loader.data(maxlen = train_x.shape[-1])
    print('padded dev shape', dev_x.shape, dev_y.shape)

    unlabeled_loader = DataLoader(text=unlabeled_text, fn2idx = fn2idx_unlabeled,
            encoder=encoder, encoder_y=encoder_y, save_name='unlabeled_loader.pickle')
    print('#unlabeled', len(unlabeled_loader))
    with open('test.txt', 'w') as f:
        unlabeled_loader.dump(f, encoder_y)
