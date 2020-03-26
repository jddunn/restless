import numpy as np
import pandas as pd
from collections import defaultdict

import sys
import os
import string

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../../..")

DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
)

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import (
    Conv1D,
    Dense,
    Input,
    InputLayer,
    Flatten,
    MaxPooling1D,
    Embedding,
    merge,
    Dropout,
    LSTM,
    GRU,
    Bidirectional,
    TimeDistributed,
)
from keras.models import Model, load_model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

import nltk

# Make imports work for Docker package and when running as a script
try:
    from restless.components.utils import utils
except:
    from utils import utils
try:
    from restless.components.nlp.text_normalizer import text_normalizer
except:
    from ..text_normalizer import text_normalizer

MAX_SENTENCE_LENGTH = 100
MAX_SENTENCE_COUNT = 15
VOCABULARY_SIZE = 20000
ATTENTION_DIM = 50
GLOVE_DIMENSION_SIZE = ATTENTION_DIM  # needs same dimension
VALIDATION_SPLIT = 0.2
MAX_DOCS = 10000000  # Limit number of records to train for speed if needed

GLOVE_DATA_PATH = os.path.abspath(
    os.path.join(
        DEFAULT_DATA_PATH, "glove", "glove.6B." + str(GLOVE_DIMENSION_SIZE) + "d.txt"
    )
)
DEFAULT_TRAINING_DATA_PATH = os.path.abspath(
    os.path.join(DEFAULT_DATA_PATH, "training", "malware-dataset.csv")
)
DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(DEFAULT_DATA_PATH, "models", "default.h5")
)


class HierarchicalAttentionNetwork:
    """
    Hierarchical Attention Network implementation.
    """

    def __init__(self, **kwargs):
        try:
            self.model = load_model(
                DEFAULT_MODEL_PATH, custom_objects={"AttentionLayer": AttentionLayer}
            )
            self.model.load_weights(DEFAULT_MODEL_PATH)
        except:
            pass
        self.texts = []
        self.texts_features = []
        self.checksums = []
        self.data_train = pd.read_csv(DEFAULT_TRAINING_DATA_PATH, nrows=MAX_DOCS)
        # self.data_train = self.data_train.drop_duplicates()
        self.records = self.data_train.to_dict("records")
        self.data = np.zeros(
            (len(self.records), MAX_SENTENCE_COUNT, MAX_SENTENCE_LENGTH), dtype="int32",
        )
        self.labels = []
        self.labels_matrix = np.zeros(len(self.records), dtype="int32")
        # For now before we integrate the db load corpus from CSV
        # We should play around with combinations of features to detect malware
        self.feature_keys_dict = [
            # {"name": "e_magic", "index": 0, "tokenize": "char"},
            # {"name": "e_cblp", "index": 1, "tokenize": "char"},
            # {"name": "e_cp", "index": 2, "tokenize": "none"},
            # {"name": "e_crlc", "index": 3, "tokenize": "none"},
            # {"name": "e_cparhdr", "index": 4, "tokenize": "none"},
            {"name": "e_minalloc", "index": 5, "tokenize": "none"},
            {"name": "e_maxalloc", "index": 6, "tokenize": "none"},
            # {"name": "e_ss", "index": 7, "tokenize": "none"},
            # {"name": "e_sp", "index": 8, "tokenize": "none"},
            # {"name": "e_csum", "index": 9, "tokenize": "none"},
            {"name": "e_ip", "index": 10, "tokenize": "char"},
            # {"name": "e_cs", "index": 11, "tokenize": "none"},
            # {"name": "e_lfarlc", "index": 12, "tokenize": "none"},
            # {"name": "e_ovno", "index": 13, "tokenize": "none"},
            # {"name": "e_res", "index": 14, "tokenize": "none"},
            # {"name": "e_oemid", "index": 15, "tokenize": "char"},
            # {"name": "e_oeminfo", "index": 16, "tokenize": "char"},
            # {"name": "e_res2", "index": 17, "tokenize": "none"},
            # {"name": "e_lfanew", "index": 18, "tokenize": "none"},
            {"name": "Machine", "index": 19, "tokenize": "none"},
            {"name": "NumberOfSections", "index": 20, "tokenize": "none"},
            {"name": "PointerToSymbolTable", "index": 22, "tokenize": "none"},
            {"name": "NumberOfSymbols", "index": 23, "tokenize": "none"},
            {"name": "AddressOfEntryPoint", "index": 32, "tokenize": "char"},
            {"name": "CheckSum", "index": 46, "tokenize": "char"},
        ]

        self.word_embedding = None
        self.word_attention_model = None
        self.tokenizer = None
        self.class_count = 2

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

        self.word_index = None
        self.embeddings_index = None
        self.embeddings_matrix = None

        self.preprocess_data(self.data_train)
        return

    def load_model(self, filepath: str):
        """Loads a model with a custom AttentionLayer property."""
        res = load_model(
            filepath, custom_objects={"AttentionLayer": AttentionLayer(Layer)}
        )
        if res:
            self.model = res
            return True
        else:
            return None

    def read_and_train_data(self, filepath: str):
        """Reads a CSV file into training data and trains network."""
        data_train = pd.read_csv(filepath, nrows=MAX_DOCS)
        # data_train = data_train.drop_duplicates()
        samples = self.preprocess_data(data_train)
        self.prepare_training_and_validation_data(samples)
        self.embeddings_index = self.get_glove_embeddings()
        embeddings_matrix = self.make_embeddings_matrix(self.embeddings_index)
        self.build_network_and_train_model(embeddings_matrix)
        return

    def preprocess_data(self, data_train: object):
        """Preprocesses data given a df object."""
        self.feature_vecs = self.build_features_vecs_from_data(
            data_train, self.feature_keys_dict
        )
        print("Total %s unique tokens." % len(self.word_index))
        print("Shape of data tensor before: ", self.data)
        samples = int(VALIDATION_SPLIT * self.data.shape[0])
        return samples

    def prepare_training_and_validation_data(self, samples):
        self.x_train = self.data[:-samples]
        self.y_train = self.labels_matrix[:-samples]
        self.x_val = self.data[-samples:]
        self.y_val = self.labels_matrix[-samples:]
        print(self.y_train.sum(axis=0))
        print(self.y_val.sum(axis=0))
        return

    def get_glove_embeddings(self, glove_data_path: str = None):
        embeddings_index = {}
        f = None
        if not glove_data_path:
            glove_data_path = GLOVE_DATA_PATH
        f = open(GLOVE_DATA_PATH)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
        f.close()
        print("Total %s word vectors." % len(embeddings_index))
        return embeddings_index

    def make_embeddings_matrix(self, embeddings_index):
        embeddings_matrix = np.random.random((len(self.word_index) + 1, ATTENTION_DIM))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embeddings_matrix[i] = embedding_vector
            else:
                # Randomly initialize vector
                embeddings_matrix[i] = np.random.normal(
                    scale=0.6, size=(ATTENTION_DIM,)
                )
        self.embeddings_matrix = embeddings_matrix
        return embeddings_matrix

    def build_network_and_train_model(
        self, embeddings_matrix, model_filepath: str = None
    ):
        """Trains a model and saves to a given filepath (will default
           to a filename)."""
        if model_filepath is None:
            model_filepath = DEFAULT_MODEL_PATH
        embedding_layer = Embedding(
            len(self.word_index) + 1,
            ATTENTION_DIM,
            weights=[self.embeddings_matrix],
            input_length=MAX_SENTENCE_LENGTH,
            trainable=True,
            mask_zero=True,
        )
        sentence_input = Input(shape=(MAX_SENTENCE_LENGTH,), dtype="int32")
        embedded_sequences = embedding_layer(sentence_input)
        l_lstm = Bidirectional(GRU(ATTENTION_DIM, return_sequences=True))(
            embedded_sequences
        )
        l_att = AttentionLayer(ATTENTION_DIM)(l_lstm)
        sentEncoder = Model(sentence_input, l_att)

        checksum_input = Input(
            shape=(MAX_SENTENCE_COUNT, MAX_SENTENCE_LENGTH), dtype="int32"
        )
        checksum_encoder = TimeDistributed(sentEncoder)(checksum_input)
        l_lstm_sent = Bidirectional(GRU(ATTENTION_DIM, return_sequences=True))(
            checksum_encoder
        )
        l_att_sent = AttentionLayer(ATTENTION_DIM)(l_lstm_sent)
        preds = Dense(2, activation="softmax")(l_att_sent)
        model = Model(checksum_input, preds)
        model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["acc"])
        print("Training HANN model now..")
        model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_val, self.y_val),
            epochs=10,
            batch_size=32,
        )
        model.save(model_filepath)
        self.model = model
        return model_filepath

    def _fill_feature_vec(self, texts_features: list, feature_vector):
        """Helper function to build feature vector for HANN to classify."""
        for i, sentences in enumerate(texts_features):
            for j, sentence in enumerate(sentences):
                if j < MAX_SENTENCE_COUNT:
                    k = 0
                    if type(sentence) is list:
                        wordTokens = [text_to_word_sequence(seq) for seq in sentence]
                    else:
                        wordTokens = text_to_word_sequence(sentence)
                    for _, word in enumerate(wordTokens):
                        if type(word) is list:
                            word = "".join(word)
                            # if (
                            # k < MAX_SENTENCE_LENGTH
                            # and
                            # self.tokenizer.word_index[word] < VOCABULARY_SIZE
                            # ):
                        feature_vector[i, j, k] = self.word_index[word]
                        k = k + 1
        return feature_vector

    def build_features_vecs_from_data(self, data_train, feature_keys_dict: dict = None):
        """Vectorizes the training dataset for HANN."""
        results = []
        if feature_keys_dict is None:
            feature_keys_dict = self.feature_keys_dict
        _data_train = data_train.to_dict()
        self.tokenizer = Tokenizer()
        self.texts = []
        self.texts_features = []
        for idx in range(
            data_train.CheckSum.shape[0]
        ):  # doesn't matter we're just getting count here
            sentences = []
            for dict in feature_keys_dict:
                key = dict["name"]
                sentence = str(_data_train[key][idx])
                sentence = text_normalizer.normalize_text(
                    sentence,
                    lowercase=True,
                    strip_punct=True,
                    # remove_stopwords=True,
                    lemmatize_text=True,
                    # stem_text=True
                )
                if dict["tokenize"] is "char":
                    sentence = [char for char in sentence]
                sentences.append(sentence)
            self.texts.extend(sentences)
            self.texts_features.append(sentences)
        self.tokenizer = Tokenizer(nb_words=VOCABULARY_SIZE)
        # self.texts = functools.reduce(operator.iconcat, self.texts, [])
        _texts = []
        for i, each in enumerate(self.texts):
            if type(each) is list:
                each = "".join(each)
            _texts.append(str(each))
        for i in range(len(self.records)):
            self.labels.append(self.records[i]["classification"])
        self.texts = _texts
        self.tokenizer.fit_on_texts(self.texts)
        self.word_index = self.tokenizer.word_index
        self.data = self._fill_feature_vec(self.texts_features, self.data)
        self.labels_matrix = to_categorical(np.asarray(self.labels))
        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)
        self.data = self.data[indices]
        self.labels_matrix = self.labels_matrix[indices]
        return self.data

    def build_features_vecs_from_input(
        self, input_features, feature_keys_dict: dict = None
    ):
        """Vectorizes a feature matrix from extracted PE data for HANN classification."""
        results = []
        if feature_keys_dict is None:
            feature_keys_dict = self.feature_keys_dict
        self.texts_features = []
        for idx in range(len(input_features)):
            sentences = []
            for dict in self.feature_keys_dict:
                val = dict["index"]
                sentence = str(input_features[val])
                sentence = text_normalizer.normalize_text(
                    sentence,
                    lowercase=True,
                    strip_punct=True,
                    # remove_stopwords=True,
                    lemmatize_text=True,
                    # stem_text=True
                )
                if dict["tokenize"] is "char":
                    sentence = [char for char in sentence]
                sentences.append(sentence)
            self.texts_features.append(sentences)
        feature_vector = np.zeros(
            (len(input_features), MAX_SENTENCE_COUNT, MAX_SENTENCE_LENGTH),
            dtype="int32",
        )
        feature_vector = self._fill_feature_vec(self.texts_features, feature_vector)
        return feature_vector

    def predict(self, data):
        """Predicts binary classification of classes with probabilities given a feature matrix."""
        res = self.model.predict(data)
        classes = to_categorical(res)
        probs = res[0]
        malicious = probs[1]
        benign = probs[0]
        return (benign, malicious)


class AttentionLayer(Layer):
    """
    Attention layer for Hierarchical Attention Network.
    """

    def __init__(self, attention_dim=ATTENTION_DIM, **kwargs):
        self.init = initializers.get("normal")
        self.supports_masking = False
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
    hann.read_and_train_data(DEFAULT_TRAINING_DATA_PATH)
else:
    utils.print_logm("Initializing HANN.")
    hann = HierarchicalAttentionNetwork()
    if hann.load_model(DEFAULT_MODEL_PATH):
        print("Succesfully loaded HANN model: ", DEFAULT_MODEL_PATH)
    else:
        hann.read_and_train_data(DEFAULT_TRAINING_DATA_PATH)
