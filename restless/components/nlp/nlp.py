import os, sys

# Silence TF warnings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import pandas as pd

# make dep imports work when running in dir and in outside scripts
PACKAGE_PARENT = "../../.."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
try:
    from restless.components.nlp.hann.hann import (
        HierarchicalAttentionNetwork,
        DEFAULT_TRAINING_DATA_PATH,
    )
    from restless.components.utils import utils as utils
except:
    from hann import HierarchicalAttentionNetwork, DEFAULT_TRAINING_DATA_PATH
    from ..utils import utils as utils

stats = utils.stats

# Get our X features to load into model
df = pd.read_csv(DEFAULT_TRAINING_DATA_PATH)
features = [x for x in list(df.columns) if x != "classification"]
target_feature = "classification"
corr = stats.get_correlation_for_features(
    df, features, target_feature=target_feature, get_corr_with_target_feature_only=True
)

top_df, top_features = stats.transform_df_with_top_features_for_hann(
    df, corr, features, target_feature, threshold=0.1
)

top_features = []


class NLP:
    """
    Module with all NLP and text processing related features.
    """

    def __init__(self, load_default_hann_model=False):
        self.load_default_hann_model = load_default_hann_model
        # We don't have "sentences" or "words" for PE header data,
        # so tokenize every string into a word
        # For example, "4069" will be considered a sentence,
        # tokenized as a sequence of words "4", "0", "6", "9".
        hann = HierarchicalAttentionNetwork(
            load_default_model=load_default_hann_model,
            features=top_features,
            word_token_level="char",
            sent_token_level="sent",
        )
        self.hann = hann
