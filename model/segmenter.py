import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class Segmenter(tf.keras.Model):
    def __init__(self, max_sentences, bidirectional, num_classification_layers):
        super(Segmenter, self).__init__(name='Segmenter')
        self.max_sentences = max_sentences
        self.num_classification_layers = num_classification_layers
        self.embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-large/5", output_shape=[512], input_shape=[], dtype=tf.string)
        
        if bidirectional:
            print('Bidirectional recurrent layer')
            self.recurrent = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, input_shape=(None, 512), return_sequences=True))
        else:
            self.recurrent = tf.keras.layers.GRU(256, input_shape=(None, 512), return_sequences=True)
        
        if self.num_classification_layers == 1:
            self.classification = tf.keras.layers.Dense(2, activation='softmax')
        else:
            self.classification_layers = []
            for i in range(self.num_classification_layers-1):
                self.classification_layers.append(tf.keras.layers.Dense(128, activation='relu'))
            self.classification_layers.append(tf.keras.layers.Dense(2, activation='softmax'))

    def call(self, inputs, prepare_inputs=False):
        if prepare_inputs:
            inputs = Segmenter.prepare_inputs(inputs, self.max_sentences)
        x = tf.reshape(inputs, [-1])
        x = self.embed(x) # (batch size * max_sentences, 512)
        x = tf.reshape(x, [-1, self.max_sentences, 512]) # (batch size, num sentences, 512)
        x = self.recurrent(x) # (batch size, max_sentences, 256)
        if self.num_classification_layers == 1:
            x = self.classification(x)
        else:
            for classification_layer in self.classification_layers:
                x = classification_layer(x)
        return x

    @staticmethod
    def prepare_inputs(inputs, max_sentences, return_tf=True, pad=''):
        x = []
        for sentences in inputs:
            x.append(sentences[:max_sentences] + [pad]*(max_sentences-len(sentences[:max_sentences])))
        
        if return_tf:
            return tf.constant(x)
        return x