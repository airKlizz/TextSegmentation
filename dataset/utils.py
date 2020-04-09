import sys
sys.path.append("..")
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from nltk.tokenize import sent_tokenize
import json
from tqdm import tqdm
from model.segmenter import Segmenter

def create_tf_dataset(train_path='wikinews/train.txt', max_sentences=32, test_size=0.2, batch_size=16, shuffle=10000, random_state=2020):
    X = []
    y = []
    mask = []
    with open(train_path, 'r') as f:
        data = f.read()
        for article in tqdm(data.split('\n\n'), desc="Reading train file"):
            X_ = []
            y_ = []
            sentences = article.split('\n')
            for sentence in sentences[1:]:
                elems = sentence.split('\t')
                assert len(elems) == 4
                X_.append(elems[-2])
                if int(elems[-1]) == 1:
                    y_.append([0, 1])
                else:
                    y_.append([1, 0])
            X.append(X_)
            y.append(y_)
            mask.append([1]*len(y_))

    X = Segmenter.prepare_inputs(X, max_sentences, return_tf=False)
    y = Segmenter.prepare_inputs(y, max_sentences, return_tf=False, pad=[1, 0])
    mask = Segmenter.prepare_inputs(mask, max_sentences, return_tf=False, pad=0)

    assert len(X) == len(y) and len(X) == len(mask)
    assert len(X[0]) == len(X[1])
    assert type(X[0][0]) == str
    assert type(y[0][0]) == list
    assert type(mask[0][0]) == int

    train_X, validation_X, train_y, validation_y, train_mask, validation_mask = train_test_split(X, y, mask, random_state=random_state, test_size=test_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y, train_mask)).shuffle(shuffle).batch(batch_size)
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_X, validation_y, validation_mask)).batch(batch_size)
    return train_dataset, validation_dataset, len(train_y)+1, len(validation_y)+1