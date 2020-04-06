# Hierarchical attention model code adapted from
# https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py,
# which implemented this https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf.

# The architecture from the paper has been modified in this model
# to accept a vector (array) as well as scalars for both the words / sentences
# that comprise a document's vocabulary. This allows us to represent any arbitrary set
# of features as a document that the HAN model can learn from, as GloVe will have
# have weights for words as well as numbers.

import numpy as np
import pandas as pd

import sys
import os
import string
import math

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

from sklearn.model_selection import KFold

from sklearn.preprocessing import RobustScaler

from sklearn.utils.multiclass import type_of_target

scaler = RobustScaler()

from sklearn.feature_extraction.text import CountVectorizer

import nltk

# make dep imports work when running as lib / in high-levels scripts
PACKAGE_PARENT = "../../../.."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

try:
    from restless.components.utils import utils
except:
    from utils import utils
try:
    from restless.components.nlp.text_normalizer import text_normalizer
except:
    from text_normalizer import text_normalizer

# Hyperparams
MAX_SENTENCE_LENGTH = 100
MAX_SENTENCE_COUNT = 48
VOCABULARY_SIZE = 100
ATTENTION_DIM = 50
GLOVE_DIMENSION_SIZE = ATTENTION_DIM  # needs same dimension
MAX_DOCS = 1000000  # Limit number of records to train for speed if needed
BATCH_SIZE = 32
EPOCH_NUM = 3

K_NUM = 5  # Number of KFold validation groups

GLOVE_DATA_PATH = os.path.abspath(
    os.path.join(
        DEFAULT_DATA_PATH, "glove", "glove.6B." + str(GLOVE_DIMENSION_SIZE) + "d.txt"
    )
)
DEFAULT_TRAINING_DATA_PATH = os.path.abspath(
    os.path.join(DEFAULT_DATA_PATH, "training", "malware-dataset.csv")
)

DEFAULT_MODEL_DIR_PATH = os.path.abspath(os.path.join(DEFAULT_DATA_PATH, "models"))

DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(DEFAULT_MODEL_DIR_PATH, "default.h5"))

stats = utils.stats
stats_vis = utils.stats_vis

compute_steps_per_epoch = lambda x: int(math.ceil(1.0 * x / BATCH_SIZE))

kf = KFold(n_splits=K_NUM, shuffle=True, random_state=1618)

metrics = ["accuracy"]

cv = CountVectorizer()

import pickle


class HierarchicalAttentionNetwork:
    """
    Hierarchical Attention Network implementation.
    """

    def __init__(self, load_default_model: bool = False, **kwargs):
        self.model = None

        self.data_train = pd.read_csv(DEFAULT_TRAINING_DATA_PATH, nrows=MAX_DOCS)
        self.records = self.data_train.to_dict("records")

        self.data = None

        self.labels = None
        self.labels_matrix = None  # Map labels into matrix for prediction

        self.feature_keys = None  # Mappings of feature indices from dataset

        self.word_index = None
        self.embeddings_index = None
        self.word_embedding = None
        self.word_attention_model = None
        self.tokenizer = None

        self.X = None
        self.Y = None

        self.num_classes = 2  # number of classes in our model; default is binary model
        if load_default_model:
            try:
                self.model = load_model(
                    DEFAULT_MODEL_PATH,
                    custom_objects={"AttentionLayer": AttentionLayer},
                )
                self.model.load_weights(DEFAULT_MODEL_PATH)
                self.data_train = pd.read_csv(
                    DEFAULT_TRAINING_DATA_PATH, nrows=MAX_DOCS
                )
                self.preprocess_data(self.data_train)
                self.embeddings_index = self.get_glove_embeddings()
                embeddings_matrix = self.make_embeddings_matrix(self.embeddings_index)
            except Exception as e:
                print("Error loading model: ", e)
        return

    def read_and_train_data(
        self,
        filepath: str,
        top_features: list = None,
        labels: list = None,
        model_base: object = None,
        outputpath: str = DEFAULT_MODEL_PATH,
        save_model: bool = False,
    ):
        """Reads a CSV file into training data and trains network."""
        self.X = pd.read_csv(filepath, nrows=MAX_DOCS)
        self.X = self.X.filter(top_features)
        # Get rid of the classification / class column since that's not a feature
        self.feature_keys = self._get_feature_keys(filepath, top_features=top_features)
        self.preprocess_data(self.X)
        self.embeddings_index = self.get_glove_embeddings()
        embeddings_matrix = self.make_embeddings_matrix(self.embeddings_index)
        model = self.build_network_and_train_model(
            embeddings_matrix, labels=labels, model_base=model_base
        )
        self.model = model
        outputpath = os.path.abspath(os.path.join(DEFAULT_MODEL_DIR_PATH, "model.p"))
        if save_model:
            self.save_model(model, outputpath)
        return model

    def preprocess_data(self, data_train: object, feature_keys: dict = None):
        """Preprocesses data given a df object."""
        self.data = np.zeros(
            (len(self.X), MAX_SENTENCE_COUNT, MAX_SENTENCE_LENGTH), dtype="int32",
        )
        self.labels_matrix = np.zeros((self.num_classes,), dtype="int32")
        if feature_keys is None:
            if self.feature_keys is None or len(self.feature_keys) is 0:
                self.feature_keys = self._get_feature_keys()
            feature_keys = self.feature_keys
        else:
            self.feature_keys = feature_keys
        self.feature_vecs = self.build_features_vecs_from_data(data_train, feature_keys)
        print("Total %s unique tokens." % len(self.word_index))
        print("Shape of data tensor: ", self.data, self.data.shape)
        return self.data

    def load_model(self, filepath: str):
        """Loads a model with a custom AttentionLayer property."""
        # model = load_model(
        #  filepath, custom_objects={"AttentionLayer": AttentionLayer(Layer)}
        # )
        if model:
            self.model = model
            return model
        else:
            return None

    def save_model(self, model: object, fp: str) -> bool:
        """
        Saves a model to a given filepath.
        """
        try:
            model.save(fp)
            return True
        except:
            return False

    def get_glove_embeddings(self, glove_data_path: str = None):
        """Use pre-trained GloVe embeddings so we don't have to
           make our own for the model.
        """
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
        """Get our word vectors for our embeddings indices."""
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
        return embeddings_matrix

    def create_model_base(self, embeddings_matrix):
        """Creates the base hierarchical attention network with multiclass
           or binary tuning. If doing non-binary classification, set
           self.num_classes to number of classes.
        """
        embedding_layer = Embedding(
            len(self.word_index) + 1,
            ATTENTION_DIM,
            weights=[embeddings_matrix],
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

        input_layer = Input(
            shape=(MAX_SENTENCE_COUNT, MAX_SENTENCE_LENGTH), dtype="int32"
        )
        layer_encoder = TimeDistributed(sentEncoder)(input_layer)
        l_lstm_sent = Bidirectional(GRU(ATTENTION_DIM, return_sequences=True))(
            layer_encoder
        )
        l_att_sent = AttentionLayer(ATTENTION_DIM)(l_lstm_sent)
        if self.num_classes > 2:
            # multi-class classifier
            preds = Dense(self.num_classes, activation="softmax")(l_att_sent)
            model = Model(input_layer, preds)
            model.compile(
                loss="categorical_crossentropy", optimizer="rmsprop", metrics=metrics
            )
        else:
            # binary classifier
            preds = Dense(2, activation="sigmoid")(l_att_sent)
            model = Model(input_layer, preds)
            model.compile(
                loss="binary_crossentropy", optimizer="rmsprop", metrics=metrics
            )
        return model

    def build_network_and_train_model(
        self,
        embeddings_matrix,
        model_base: object = None,
        model_filepath: str = None,
        labels: list = ["0", "1"],
        save_metrics_results: bool = False,
    ):
        """Trains a model and saves to a given filepath (will default
           to a filename)."""
        if not model_base:
            model = self.create_model_base(embeddings_matrix)
            print("Creating HANN architecture with Keras - {}.".format(model))
        else:
            print("Creating non-HANN model base {}.".format(model_base))
            model = model_base
        k_ct = 1  # Which k-index are we on in kfold val
        metrics_arr = []
        models = []  # store all our models and we'll save the best performing one
        # Kfold validation
        for train_index, test_index in kf.split(self.X, self.Y):
            x_train, x_val = self.X[train_index], self.X[test_index]
            y_train, y_val = self.Y[train_index], self.Y[test_index]
            if not model_base:
                # We're creating a HANN model using Keras
                print(
                    "Creating HANN model now, with K-Fold cross-validation. K=",
                    k_ct,
                    "and length: ",
                    len(x_train),
                    len(x_val),
                    "for training / validation.",
                )
                model.fit(
                    x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    epochs=EPOCH_NUM,
                    batch_size=BATCH_SIZE,
                    verbose=2,
                )
            else:
                # If we're given a model base, then we're probably
                # not building a HANN model, but this class is being
                # used to create baseline models for comparison with HANN.
                # This really should be refactored out into another class, but
                # for now this is clean enough.

                # Reshape the data from 3D to 2D (for non-HANN models)
                x_val = x_val.reshape(len(x_val), -1)
                x_train = x_train.reshape(len(x_train), -1)
                x_train = scaler.fit_transform(x_train)
                x_val = scaler.fit_transform(x_val)
                y_train = np.argmax(y_train, axis=1)
                print(
                    "Creating baseline model now, with K-Fold cross-validation. K=",
                    k_ct,
                    "and length: ",
                    len(x_train),
                    len(x_val),
                    "for training / validation.",
                )
                model.fit(x_train, y_train)
            models.append(model)
            # f_importances(model.coef_, feature_keys_list)
            k_ct += 1
            y_pred = model.predict(x_val)
            # Reverse one-hot encoding
            _y_val = np.argmax(y_val, axis=1)
            if not model_base:
                _y_pred = np.argmax(y_pred, axis=1)
            else:
                _y_pred = y_pred
            # Model evaluation / metrics
            metrics = stats.get_model_metrics(
                _y_val, _y_pred, labels, print_output=True
            )
            metrics_arr.append(metrics)
        # Drop confusion matrices from our summed metrics since we can't average those easily
        filtered_metrics_arr = [
            {k: v for k, v in d.items() if k != "cm"} for d in metrics_arr
        ]
        metrics_summed = stats.get_metrics_averages(filtered_metrics_arr)
        print("Metrics summed and averaged: ", metrics_summed)
        top_score = 0
        index = 1
        for i, metrics in enumerate(metrics_arr):
            # We'll save the model with the best performing metric (F1 score)
            if metrics["f1"] > top_score:
                top_score = metrics["f1"]
                if i is 0:
                    index = i + 1
                else:
                    index = i
        best_model = models[index]
        model = best_model
        print(
            "The best performing model (based on F1 score) was number {}. That is the model that will be returned.".format(
                index
            )
        )
        if save_metrics_results:
            pass
        return model

    def _get_feature_keys(
        self, filepath: str = DEFAULT_TRAINING_DATA_PATH, top_features: list = None
    ) -> list:
        if self.feature_keys is None or len(self.feature_keys) is 0:
            df = pd.read_csv(filepath, nrows=MAX_DOCS)
            if top_features is not None and len(top_features) is not 0:
                feature_keys = [
                    key
                    for key in list(df.columns)
                    if key != "classification" and key in top_features
                ]
            else:
                feature_keys = [
                    key for key in list(df.columns) if key != "classification"
                ]
            # Map feature keys with their indices (since eventually we may want to eliminate features
            # from being trained, without modifiying the original dataset, so order of indices may not
            # be continuous. Also, we can define a tokenization level in these mappings.
            feature_keys = [
                {"name": feature_key, "index": i}
                for i, feature_key in enumerate(feature_keys)
            ]
            return feature_keys
        else:
            return self.feature_keys

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
                        if k < MAX_SENTENCE_LENGTH and _ < VOCABULARY_SIZE:
                            try:
                                feature_vector[i, j, k] = self.word_index[word]
                            except:
                                feature_vector[i, j, k] = 0
                            k = k + 1
                        else:
                            break
        return feature_vector

    def build_features_vecs_from_data(self, data_train, feature_keys: dict = None):
        """Vectorizes the training dataset for HANN."""
        if feature_keys is None:
            if self.feature_keys is None or len(self.feature_keys) is 0:
                self.feature_keys = self._get_feature_keys()
            feature_keys = self.feature_keys
        else:
            self.feature_keys = feature_keys
        results = []
        data_train_dict = data_train.to_dict()
        self.tokenizer = Tokenizer()
        texts = []
        texts_features = []
        self.labels = []
        for idx in range(
            data_train.shape[0]
        ):  # doesn't matter we're just getting count here
            sentences = []
            for dict in feature_keys:
                key = dict["name"]
                sentence = str(data_train[key][idx])
                sentence = text_normalizer.normalize_text_defaults(sentence)
                if "tokenize" in dict and dict["tokenize"] is "char":
                    sentence = [char for char in sentence]
                sentences.append(sentence)
            texts.extend(sentences)
            texts_features.append(sentences)
        self.tokenizer = Tokenizer(nb_words=VOCABULARY_SIZE)
        _texts = []
        for i, each in enumerate(texts):
            if type(each) is list:
                each = "".join(each)
            _texts.append(str(each))
        for i in range(len(self.records)):
            self.labels.append(self.records[i]["classification"])
        texts = _texts
        self.tokenizer.fit_on_texts(texts)
        self.word_index = self.tokenizer.word_index
        self.data = self._fill_feature_vec(texts_features, self.data)
        # Get unique classes from labels (in same order of occurence)
        classes = [x for i, x in enumerate(self.labels) if self.labels.index(x) == i]
        self.num_classes = len(classes)
        # Shuffle data
        # indices = np.arange(self.data.shape[0])
        # np.random.shuffle(indices)
        # self.data = self.data[indices]
        # self.labels = self.labels[indices]
        self.labels_matrix = to_categorical(self.labels, num_classes=self.num_classes)
        print(self.labels_matrix, self.labels_matrix.shape)
        self.Y = self.labels_matrix
        self.X = self.data
        return self.data

    def build_features_vecs_from_input(self, input_features, feature_keys: dict = None):
        """Vectorizes a feature matrix from extracted PE data for HANN classification."""
        results = []
        texts_features = []
        if feature_keys is None:
            if self.feature_keys is None or len(self.feature_keys) is 0:
                self.feature_keys = self._get_feature_keys()
            feature_keys = self.feature_keys
        else:
            self.feature_keys = feature_keys
        for idx in range(len(input_features)):
            sentences = []
            for dict in self.feature_keys:
                val = dict["index"]
                sentence = str(input_features[val])
                sentence = text_normalizer.normalize_text(sentence)
                if "tokenize" in dict and dict["tokenize"] is "char":
                    sentence = [char for char in sentence]
                sentences.append(sentence)
            texts_features.append(sentences)
        feature_vector = np.zeros(
            (len(input_features), MAX_SENTENCE_COUNT, MAX_SENTENCE_LENGTH),
            dtype="int32",
        )
        feature_vector = self._fill_feature_vec(texts_features, feature_vector)
        # feature_vector = feature_vector.reshape(len(feature_vector), -1)
        return feature_vector

    def predict(self, data):
        """Predicts binary classification of classes with probabilities given a feature matrix."""
        res = self.model.predict(data)
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
        # Masking layers is no longer supported in newer version of keras
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
    print("Use train_hann.py to train the HANN network, or import as a module.")
else:
    utils.print_logm("Initializing HANN module.")
