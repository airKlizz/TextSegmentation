import sys
sys.path.append("..")
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from nltk.tokenize import sent_tokenize
import json
from tqdm import tqdm
from model.segmenter import Segmenter

def create_tf_dataset(train_path='wikinews/data.jsonl', max_sentences=32, test_size=0.2, batch_size=16, shuffle=10000, random_state=2020):
    X = []
    y = []
    with open(train_path, 'r') as f:
        for line in tqdm(f, desc="Reading train file"):
            json_line = json.loads(line)
            passages = json_line['passages']
            X_ = []
            y_ = []
            for passage in passages:
                sentences = sent_tokenize(passage)
                X_ += sentences
                y_ += [[0, 1]]+[[1, 0]]*(len(sentences)-1)
            X.append(X_)
            y.append(y_)

    X = Segmenter.prepare_inputs(X, max_sentences, return_tf=False)
    y = Segmenter.prepare_inputs(y, max_sentences, return_tf=False, pad=[1, 0])

    train_X, validation_X, train_y, validation_y = train_test_split(X, y, random_state=random_state, test_size=test_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).shuffle(shuffle).batch(batch_size)
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_X, validation_y)).batch(batch_size)
    return train_dataset, validation_dataset, len(train_y)+1, len(validation_y)+1