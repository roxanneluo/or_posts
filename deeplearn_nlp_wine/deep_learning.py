from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from lib.get_top_xwords import filter_to_top_x
from prepare_dataset import *
from encoder import Encoder, LabelEncoder
import tensorflow as tf
from pdb import set_trace as st
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('percent', default=0, type=float)
args = parser.parse_args()

percent = args.percent
np.random.seed(0)
tf.set_random_seed(0)

embedding_vector_length = 64

encoder = Encoder()
encoder_y = LabelEncoder()
n_labels = len(encoder_y.vocab())

train_loader = DataLoader(save_name='train.pickle')
train_init_x, train_init_y = train_loader.data(num_categories = n_labels)
dev_loader = DataLoader(save_name='dev.pickle')
test_x, test_y = dev_loader.data(num_categories = n_labels, maxlen = train_init_x.shape[-1])
## unlabeled
unlabeled_loader = DataLoader(save_name='unlabeled_loader.pickle')
unlabeled_x, unlabeled_y = unlabeled_loader.data(num_categories = n_labels, maxlen = train_init_x.shape[-1])
n_unlabeled = unlabeled_x.shape[0]

model = Sequential()

model.add(Embedding(len(encoder.vocab()), embedding_vector_length, input_length=train_init_x.shape[-1]))
model.add(Conv1D(50, 5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(len(encoder_y.vocab()), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_x, train_y = train_init_x, train_init_y
for t in range(2):
    model.fit(train_x, train_y, epochs=5, batch_size=64)

    y_score = model.predict(test_x)
    y_score = [[1 if i == max(sc) else 0 for i in sc] for sc in y_score]
    n_right = 0
    for i in range(len(y_score)):
        if all(y_score[i][j] == test_y[i][j] for j in range(len(y_score[i]))):
            n_right += 1

    print("Accuracy: %.2f%%" % ((n_right/float(len(test_y)) * 100)))

    if percent > 0:
        idxs = np.random.choice(n_unlabeled, size = int(percent * n_unlabeled))
        unlabeled_x_samples = unlabeled_x[idxs]
        unlabeled_y_samples = model.predict(unlabeled_x_samples)
        train_x, train_y = np.concatenate((train_init_x, unlabeled_x_samples), axis=0), np.concatenate((train_init_y, unlabeled_y_samples), axis=0)

unlabeled_y = model.predict(unlabeled_x)
unlabeled_loader.y = np.argmax(unlabeled_y, axis=1)
with open('unlabeled_result_%f.txt' % percent, 'w') as f:
    unlabeled_loader.dump(f, encoder_y)
