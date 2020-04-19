# Hierarchical attention model code adapted from
# https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py,
# which implemented this https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf.

# Here we use the HANN model with data that follows a structured format (extracted from a CSV dataset
# of portable executable header data). If we consider each feature of the dataset to correspond
# to a `sentence`, and thus the tokens of the features to comprise `words` in our model vocab,
# can we build arbitrary reprensentations of documents for classification for HANN's architecture?
# And, if we keep the order of our training and input data consistent, our model should actually learn
# which features matter the most and focus on those, and also be able to visualize this hierarchy
# using HANN's attention maps.

# The architecture from the paper has been modified in this model to custom levels of tokenization
# at the word and sentence level, which allows us to represent any arbitrary set
# of features as a document that the HAN model can learn from, as GloVe will have
# have weights for words as well as numbers.

# We can also pass in a list of features to the class instance (that should be mapped to the columns
# in the training CSV file). If the class is instantiated with features, models will be trained
# with only those column headers / features, otherwise, all columns will be taken.
# The $feature_map will be built automatically from $features, but can also be manually passed in.
# See the example in ".." on the data structure, which follows this form:
# { "name": "CheckSum", "index": "42", "tokenization": "char" }, where name corresponds to the feature
# / column name, index with the column index, and tokenization (optional, defaulting to whitespace tokenization)
# being the custom tokenization level for this feature.

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

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
from keras.models import Model, load_model, Sequential

from sklearn.model_selection import KFold

from sklearn.preprocessing import RobustScaler

from sklearn.utils.multiclass import type_of_target

scaler = RobustScaler()

import nltk

from attention import AttentionLayer, ATTENTION_DIM

import pickle  # Once we train the model we'll load the corpus / word index

# as serialized objs so we don't have to preprocess in prediction

# make dep imports work when running as lib / in high-levels scripts
PACKAGE_PARENT = "../../../.."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

try:
    from restless.components.utils import utils
    from restless.components.nlp.text_normalizer import text_normalizer
except:
    from utils import utils
    from text_normalizer import text_normalizer

# Hyperparams
MAX_SENTENCE_LENGTH = 100
MAX_SENTENCE_COUNT = 48
GLOVE_DIMENSION_SIZE = ATTENTION_DIM  # needs same dimension
MAX_DOCS = 1000000  # Limit number of records to train for speed if needed
MAX_WORDS = 100000  # Max number of words for our corpus
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

# Pickled objects to load when we load models
DEFAULT_MODEL_ASSETS_PATH = os.path.abspath(
    os.path.join(DEFAULT_MODEL_DIR_PATH, "model_assets")
)

stats = utils.stats
stats_vis = utils.stats_vis
misc = utils.misc

kf = KFold(n_splits=K_NUM, shuffle=True, random_state=1618)

metrics = ["accuracy"]


class HierarchicalAttentionNetwork:
    """
    Hierarchical Attention Network implementation.
    """

    def __init__(
        self,
        load_default_model: bool = False,
        features: list = [],  # List of features
        word_token_level: str = "word",  # Default tokenization level for words
        sent_token_level: str = "sent",  # Default tokenization level for sentences
        **kwargs
    ):
        self.model = None
        self.model_name = ""

        self.data_train = pd.read_csv(
            DEFAULT_TRAINING_DATA_PATH, nrows=MAX_DOCS
        )  # Training data
        self.records = self.data_train.to_dict(
            "records"
        )  # Same as data_train but in dict
        self.data = None

        self.texts = None  # will become X
        self.labels = []  # will become Y
        self.labels_matrix = None  # Map labels into matrix for prediction

        self.word_index = None
        self.embeddings_index = None
        self.word_embedding = None
        self.word_attention_model = None
        self.tokenizer = None
        self.vocab_size = None

        self.X = None
        self.Y = None

        self.features = features  # List of features to extract
        self.feature_map = []  # Mappings of feature indices from dataset (will be
        # automatically done with default indices and
        # tokenization levels, see docs for more details)

        self.num_classes = 2  # number of classes in our model; default is binary model

        # By defining custom tokenizations at the word and sentence level, we can create
        # arbitrary representations of documents using things like metadata and numbers
        self.word_token_level = word_token_level
        self.sent_token_level = sent_token_level

        if load_default_model:
            self.model = load_model(
                DEFAULT_MODEL_PATH,
                custom_objects={"AttentionLayer": AttentionLayer},
                compile=False,
            )
            self.model_name = DEFAULT_MODEL_PATH.split("/")[
                len(DEFAULT_MODEL_PATH.split("/")) - 1
            ]
            # Get the original training vocabulary (should load from file / db later)
            self.data_train = pd.read_csv(DEFAULT_TRAINING_DATA_PATH, nrows=MAX_DOCS)
            if len(self.feature_map) is 0:
                # Automatically map our features from CSV columns (if not manually specified)
                self.feature_map = self._get_feature_map(
                    DEFAULT_TRAINING_DATA_PATH, top_features=self.features
                )
            self.preprocess_data(
                self.data_train,
                self.feature_map,
                word_token_level=self.word_token_level,
                sent_token_level=self.sent_token_level,
            )
            print(
                "Successfully loaded default HANN model - {} - {}.".format(
                    DEFAULT_MODEL_PATH, self.model_name
                )
            )
        return

    def read_and_train_data(
        self,
        filepath: str,
        top_features: list = None,
        labels: list = None,
        model_base: object = None,
        outputpath: str = DEFAULT_MODEL_PATH,
        save_model: bool = False,
        word_token_level: str = "word",
        sent_token_level: str = "sent",
    ):
        """Reads a CSV file into training data and trains network."""
        self.X = pd.read_csv(filepath, nrows=MAX_DOCS)
        if top_features is not None:
            self.X = self.X.filter(top_features)
        else:
            if len(self.features) > 0:
                top_features = self.features
                self.X = self.X.filter(self.features)
        self.feature_map = self._get_feature_map(filepath, top_features=top_features)
        self.preprocess_data(
            self.X, self.feature_map, word_token_level, sent_token_level
        )
        print("Finished preprocessing data.")
        self.embeddings_index = self.get_glove_embeddings()
        print("Finished getting GlOve embeddings.")
        embeddings_matrix = self.make_embeddings_matrix(self.embeddings_index)
        print("Finished making embeddings matrix from word index.")
        model = self.build_network_and_train_model(
            embeddings_matrix, labels=labels, model_base=model_base
        )
        print("Finished training model.")
        self.model = model
        self.model_name = output_path.split("/")[len(output_path.split("/")) - 1]
        if save_model:
            self.save_model(model, outputpath)
            print("Finished saving model: {} at {}".format(model_name, output_path))
        return model

    def _build_corpus(
        self,
        data_train: pd.DataFrame,
        feature_map: dict,
        word_token_level: str = "word",
        sent_token_level: str = "sent",
    ):
        """Builds a corpus from a dataframe, given a dictionary of features
           mapped to their indices and tokenization level. These will be
           automatically generated if they're not individually specified.
           This allows arbitrary levels of tokenization, meaning we can
           consider any feature or sequence of text (even if it's numbers)
           as a sentence, and then we can take individual chars or groupings
           of chars to count as words for the vocab.

           Right now, by default we'll tokenize at the char level for words,
           since our default HANN model will use PE header metadata (numbers).
           Eventually though, we'll want to manually (or programmatically)
           analyze and define which metadata features we should tokenize by
           chars and which ones by other hierustics. For example, let's say
           we have a rule that states if a sequence of numbers (a "sentence")
           is divisible by 4, 8, and 16, then we should not tokenize that
           feature by individual chars, as that number probably is more meaningful
           as a whole token. Or how about for numbers that correspond to commonly
           used ports, like 80, and 8069? It may improve our model results to
           build this into the pre-processing pipeline.
        """
        data_train_dict = data_train.to_dict()
        texts = []
        texts_features = []
        for idx in range(data_train.shape[0]):
            sentences = []
            for feature in feature_map:
                key = feature["name"]
                sentence = str(data_train[key][idx])
                sentence = text_normalizer.normalize_text_defaults(sentence)
                if (
                    "tokenize" in feature and feature["tokenize"] is "sent"
                ) or sent_token_level is "sent":
                    sentence = text_normalizer.tokenize_text(
                        sentence, token_level="sent"
                    )
                elif (
                    "tokenize" in feature and feature["tokenize"] is "char"
                ) or sent_token_level is "char":
                    # We don't have "sentences" or "words" for PE header data,
                    # so tokenize every string into a word
                    # For example, "4069" will be considered a sentence,
                    # tokenized as a sequence of words "4", "0", "6", "9".
                    sentence = text_normalize.tokenize(sentence, token_level="char")
                else:
                    # by default we tokenize into sentences
                    sentence = text_normalizer.tokenize_text(
                        sentence, token_level="sent"
                    )
                sentences.append(sentence)
            texts.extend(sentences)
            texts_features.append(sentences)
        return (texts, texts_features)

    def preprocess_data(
        self,
        data_train: pd.DataFrame,
        feature_map: dict,
        word_token_level: str = "word",
        sent_token_level: str = "sent",
        pickle_data: bool = True
    ):
        """Preprocesses data given a df object."""
        self.data = np.zeros(
            (len(data_train), MAX_SENTENCE_COUNT, MAX_SENTENCE_LENGTH), dtype="int32",
        )
        self.labels_matrix = np.zeros((self.num_classes,), dtype="int32")
        # Read or write preprocessed data into pickle
        text_corpus_asset_path = self.model_name.split(".")[0] + "_text_corpus.p"
        word_index_asset_path = self.model_name.split(".")[0] + "_word_index.p"
        vectorized_data_asset_path = self.model_name.split(".")[0] + "_vectorized_data.p"
        read = misc.read_pickle_data(text_corpus_asset_path)
        if not read:
            self.texts = self._build_corpus(data_train, feature_map, word_token_level, sent_token_level)
            print("Finished building corpus.")
            if pickle_data:
                misc.write_pickle_data(self.texts, text_corpus_asset_path)
        else:
            self.texts = read
            print("Finished loading corpus.")
        read = misc.read_pickle_data(word_index_asset_path)
        if not read:
            self._build_feature_matrix_from_data(data_train, feature_map)
            print("Finished building feature matrix from corupus.")
            if pickle_data:
                misc.write_pickle_data(self.word_index, word_index_asset_path)
                misc.write_pickle_data(self.data, vectorized_data_asset_path)
        else:
            self.word_index = read
            read = misc.read_pickle_data(vectorized_data_asset_path)
            if not read:
                self._build_feature_matrix_from_data(data_train, feature_map)
                print("Finished building feature matrix from corupus.")
            else:
                self.data = read
        print("Total %s unique tokens." % len(self.word_index))
        print("Shape of data tensor: ", self.data, self.data.shape)
        print("Finished preprocessing data.")
        return self.data

    def predict(self, data):
        """Predicts binary classification of classes with probabilities given a feature matrix."""
        res = self.model.predict(data)
        probs = res[0]
        normal = probs[0]  # "0" class
        deviant = probs[1]  # "1" class
        attention_vals = res[1]
        return (normal, deviant)

    def load_model(self, filepath: str):
        """Loads a model with a custom AttentionLayer property."""
        model = load_model(
            filepath, custom_objects={"AttentionLayer": AttentionLayer(Layer)}
        )
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
        print("Total %s word vectors from GloVe." % len(embeddings_index))
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
            model = Model(input_layer)
            model = Model(input_layer, preds)
            model.compile(
                loss="categorical_crossentropy", optimizer="adadelta", metrics=metrics
            )
        else:
            # binary classifier
            preds = Dense(2, activation="softmax")(l_att_sent)
            model = Model(input_layer, preds)
            model.compile(
                loss="binary_crossentropy", optimizer="adadelta", metrics=metrics
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

    def get_attention_map(self, input_data):
        att_model_output = self.model.layers[0:-2]
        att_model = Model(att_model_output[0].input, att_model_output[-1].output)
        att_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return att_model.predict(input_data)[1]

    def build_feature_matrix_from_input_arr(
        self, input_features, feature_map: dict = None
    ):
        """Vectorizes a feature matrix from extracted PE data for HANN classification."""
        results = []
        texts_features = []
        if feature_map is None:
            if self.feature_map is None:
                self.feature_map = self._get_feature_map()
            feature_map = self.feature_map
        else:
            self.feature_map = feature_map
        for idx in range(len(input_features)):
            sentences = []
            for dict in self.feature_map:
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
        return feature_vector

    def _build_feature_matrix_from_data(
        self, data_train: pd.DataFrame, feature_map: dict = None
    ):
        """Vectorizes the training dataset for HANN."""
        results = []
        self.texts, texts_features = self._build_corpus(data_train, feature_map)
        self.data = self._fill_feature_vec(texts_features, self.data)
        # Get unique classes from labels (in same order of occurence)
        classes = [x for i, x in enumerate(self.labels) if self.labels.index(x) == i]
        self.num_classes = len(classes)
        self.labels_matrix = to_categorical(self.labels, num_classes=self.num_classes)
        self.Y = self.labels_matrix
        self.X = self.data
        return self.data

    def _get_feature_map(
        self, filepath: str = DEFAULT_TRAINING_DATA_PATH, top_features: list = []
    ) -> list:
        df = pd.read_csv(filepath, nrows=MAX_DOCS)
        feature_map = []
        if len(top_features) > 0:
            feature_map = [
                key
                for key in list(df.columns)
                if key != "classification" and key in top_features
            ]
        else:
            if len(self.feature_map) is 0:
                if len(self.features) > 0:
                    feature_map = [
                        key
                        for key in list(df.columns)
                        if key != "classification" and key in self.features
                    ]
                else:
                    feature_map = [
                        key for key in list(df.columns) if key != "classification"
                    ]
            else:
                return self.feature_map
            # Map feature keys with their indices (since eventually we may want to eliminate features
            # from being trained, without modifiying the original dataset, so order of indices may not
            # be continuous. Also, we can define a tokenization level in these mappings.
        feature_map = [
            {"name": feature_key, "index": i}
            for i, feature_key in enumerate(feature_map)
        ]
        return feature_map

    def _fill_feature_vec(self, texts_features: list, feature_vector):
        """Helper function to build feature vector for HANN to classify."""
        self.tokenizer = Tokenizer()
        _texts = []
        for i, each in enumerate(self.texts):
            if type(each) is list:
                each = "".join(each)
            _texts.append(str(each))
        for i in range(len(self.records)):
            self.labels.append(self.records[i]["classification"])
        texts = _texts
        self.texts = _texts
        self.tokenizer.fit_on_texts(self.texts)
        self.word_index = self.tokenizer.word_index
        self.vocab_size = len(self.word_index)
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
                        if k < MAX_SENTENCE_LENGTH and _ < MAX_WORDS:
                            try:
                                feature_vector[i, j, k] = self.word_index[word]
                            except Exception as e:
                                feature_vector[i, j, k] = 0
                            k = k + 1
                        else:
                            break
        return feature_vector


if __name__ == "__main__":
    print("Use train_hann.py to train the HANN network, or import as a module.")
