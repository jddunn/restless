import numpy as np
import pandas as pd
from collections import defaultdict

import sys
import os

sys.path.append("../")
sys.path.append("../../")

# Following lines are for assigning parent directory dynamically.
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import (
    Conv1D,
    MaxPooling1D,
    Embedding,
    merge,
    Dropout,
    LSTM,
    GRU,
    Bidirectional,
    TimeDistributed,
)
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

import nltk


try:
    from restless.components.utils import utils
except:
    from utils import utils
from text_normalizer import TextNormalizer

text_normalizer = TextNormalizer()

MAX_SENTENCE_LENGTH = 100
MAX_SENTENCE_COUNT = 15
VOCABULARY_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

GLOVE_DIR_PATH = "."


class HierarchicalAttentionNetwork:
    """
    Hierarchical Attention Network implementation.
    """

    def __init__(self):
        self.model = None
        self.MAX_SENTENCE_LENGTH = MAX_SENTENCE_LENGTH
        self.MAX_SENTENCE_COUNT = MAX_SENTENCE_COUNT
        self.VOCABULARY_SIZE = VOCABULARY_SIZE
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.VALIDATION_SPLIT = VALIDATION_SPLIT

        self.word_embedding = None
        self.word_attention_model = None
        self.tokenizer = None
        self.class_count = 2

        self.data = None

        self.reviews = []
        self.labels = []
        self.texts = []

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

        self.word_index = None
        self.embeddings_matrix = None
        return

    def read_data(self, filepath: str = None):
        data_train = None
        if filepath:
            data_train = pd.read_csv(filepath, sep="\t", nrows=100)
            print(data_train.shape)
        else:
            data_train = pd.read_csv("labeledTrainData.tsv", sep="\t", nrows=100)
            print(data_train.shape)
        samples, labels = self.preprocess_data(data_train)
        self.prepare_training_and_validation_data(samples, labels)
        embeddings_index = self.get_glove_embeddings()
        embeddings_matrix = self.make_embeddings_matrix(embeddings_index)
        self.build_network_and_train_model(embeddings_matrix)
        return

    def preprocess_data(self, data_train: object):
        for idx in range(data_train.review.shape[0]):
            # text = BeautifulSoup(data_train.review[idx])
            # text = clean_str(text.get_text().encode('ascii', 'ignore'))
            text = text_normalizer.normalize_text(
                data_train.review[idx],
                lowercase=True,
                strip_punct=True,
                # remove_stopwords=True,
                lemmatize_text=True,
                # stem_text=True
            )
            print(idx)
            self.texts.append(text)
            sentences = nltk.tokenize.sent_tokenize(text)
            self.reviews.append(sentences)
            self.labels.append(data_train.sentiment[idx])
        self.data = np.zeros(
            (len(self.texts), MAX_SENTENCE_COUNT, MAX_SENTENCE_LENGTH), dtype="int32"
        )
        self.tokenizer = Tokenizer(nb_words=self.VOCABULARY_SIZE)
        self.tokenizer.fit_on_texts(self.texts)
        for i, sentences in enumerate(self.reviews):
            for j, sent in enumerate(sentences):
                if j < MAX_SENTENCE_COUNT:
                    wordTokens = text_to_word_sequence(sent)
                    k = 0
                for _, word in enumerate(wordTokens):
                    if (
                        k < MAX_SENTENCE_LENGTH
                        and self.tokenizer.word_index[word] < VOCABULARY_SIZE
                    ):
                        self.data[i, j, k] = self.tokenizer.word_index[word]
                        k = k + 1
        self.word_index = self.tokenizer.word_index
        print("Total %s unique tokens." % len(self.word_index))
        self.labels = to_categorical(np.asarray(self.labels))
        print("Shape of data tensor:", self.data.shape)
        print("Shape of label tensor:", self.labels.shape)

        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)
        self.data = self.data[indices]
        self.labels = self.labels[indices]
        samples = int(VALIDATION_SPLIT * self.data.shape[0])
        return samples, self.labels

    def prepare_training_and_validation_data(self, samples, labels):
        self.x_train = self.data[:-samples]
        self.y_train = self.labels[:-samples]
        self.x_val = self.data[-samples:]
        self.y_val = labels[-samples:]
        print("Number of positive and negative reviews in traing and validation set")
        print(self.y_train.sum(axis=0))
        print(self.y_val.sum(axis=0))
        return

    def get_glove_embeddings(self, glove_dir: str = GLOVE_DIR_PATH):
        embeddings_index = {}
        f = None
        try:
            f = open(os.path.join(glove_dir, "glove.6B.100d.txt"))
        except:
            f = open(os.path.join("./components/nlp/hann", "glove.6B.100d.txt"))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
        f.close()
        print("Total %s word vectors." % len(embeddings_index))
        return embeddings_index

    def make_embeddings_matrix(self, embeddings_index):
        embeddings_matrix = np.random.random(
            (len(self.word_index) + 1, self.EMBEDDING_DIM)
        )
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embeddings_matrix[i] = embedding_vector
        self.embeddings_matrix = embeddings_matrix
        return embeddings_matrix

    def build_network_and_train_model(
        self, embeddings_matrix, model_filepath: str = None
    ):
        if model_filepath is None:
            model_filepath = "default" + ".h5"
        embedding_layer = Embedding(
            len(self.word_index) + 1,
            self.EMBEDDING_DIM,
            weights=[self.embeddings_matrix],
            input_length=self.MAX_SENTENCE_LENGTH,
            trainable=True,
            mask_zero=True,
        )

        sentence_input = Input(shape=(MAX_SENTENCE_LENGTH,), dtype="int32")
        embedded_sequences = embedding_layer(sentence_input)
        l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
        l_att = AttentionLayer(100)(l_lstm)
        sentEncoder = Model(sentence_input, l_att)

        review_input = Input(
            shape=(MAX_SENTENCE_COUNT, MAX_SENTENCE_LENGTH), dtype="int32"
        )
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
        l_att_sent = AttentionLayer(100)(l_lstm_sent)
        preds = Dense(2, activation="softmax")(l_att_sent)
        model = Model(review_input, preds)
        model.compile(
            loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"]
        )
        print("model fitting - Hierachical attention network")
        model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_val, self.y_val),
            nb_epoch=10,
            batch_size=50,
        )
        model.save(model_filepath)
        self.model = model


class AttentionLayer(Layer):
    """
    Attention layer for Hierarchical Attention Network.
    """

    def __init__(self, attention_dim):
        self.init = initializers.get("normal")
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name="W")
        self.b = K.variable(self.init((self.attention_dim,)), name="b")
        self.u = K.variable(self.init((self.attention_dim, 1)), name="u")
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttentionLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # return mask
        return None

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


if __name__ == "__main__":
    hann = HierarchicalAttentionNetwork()
    # hann.read_data("./labeledTrainData.tsv")
    hann.read_data("./malware-dataset.csv")
else:
    utils.print_logm("Initializing HANN.")
    hann = HierarchicalAttentionNetwork()
    hann.read_data("./components/nlp/hann/labeledTrainData.tsv")
    # hann.read_data("./components/nlp/hann/malware-dataset.csv")
    GLOVE_DIR_PATH = "./components/nlp/hann"
