import os, sys
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# make dep imports work when running in dir and in outside scripts
PACKAGE_PARENT = "../../../.."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
try:
    from restless.components.nlp.hann.hann import HierarchicalAttentionNetwork
    from restless.components.nlp.hann.hann import (
        DEFAULT_DATA_PATH,
        DEFAULT_TRAINING_DATA_PATH,
        DEFAULT_MODEL_PATH,
    )
    from restless.components.nlp.hann.feature_keys import *
    from restless.components.utils import utils
except:
    from hann import HierarchicalAttentionNetwork
    from hann import (
        DEFAULT_DATA_PATH,
        DEFAULT_TRAINING_DATA_PATH,
        DEFAULT_MODEL_PATH,
    )
    from feature_keys import *
    from ...utils import utils

stats = utils.stats
stats_vis = utils.stats_vis
scaler = RobustScaler()

N_FEATURES = (
    24  # Minimum number of top features to extract (based on Pearson correlation)
)


def get_features_corr(
    training_filepath: str,
    features: list,
    target_feature: str = None,
    correlation="pearson",
) -> list:
    """
    Gets correlation of each feature in a list compared to the classification
    probability of the model.

    Args:
        training_filepath (str): Filepath to read data from (should be CSV or txt)
        features (list): List of features to get correlations with
        target_feature (str, optional): If specified, will return correlation
            values for every
        correlation (str, optional): Type of correlation metric to classify;
            defaults to "jaccard".

    Returns:
        list: List of correlations; if no $target_feature param is specified, the
            list will only contain one result, with all the feature correlations.
    """
    results = []
    df = pd.read_csv(training_filepath)
    if target_feature:
        print(
            "Getting correlation for each feature with target_feature {}.".format(
                target_feature
            )
        )
        for feature in features:
            feature = [feature]
            result = {}
            corr = stats.get_correlation_for_features(
                df, feature, target_feature, correlation
            )
            # print("\tCorrelation for {} is {}.".format(feature, corr))
            result["feature"] = feature
            result["target_feature"] = target_feature
            result["corr"] = corr
            results.append(result)
    else:
        result = {}
        print("Getting correlation for all features {}.".format(features))
        corr = stats.get_correlation_for_features(df, features, correlation)
        result["features"] = features
        result["corr"] = corr
        results.append(result)
        print("\t", corr)
    return results


def train_model(
    training_fp: str,
    feature_keys: dict,
    model_base: object = None,
    model_save: bool = True,
    model_fp: str = DEFAULT_MODEL_PATH,
) -> object:
    """
    Trains a hierarchical attention network model from a CSV or text file.

    Args:
        training_fp (str): Filepath to read dataset from into df;
             must be CSV or text.
        feature_keys (dict): Dictionary containing features and their
            properties mapped from the training file.
        model_base (object, optional): If specified, train a classifier with
           given model (instead of default HANN). Should be used to test
           trained model with various baselines.
        model_save (bool, optional): Whether to save trained model to disk.
        model_fp (str, optional): Filepath to save the model to, if
            model_save is set to True. Will default if not specified.

    Returns:
        object: Trained model.
    """
    # For now the PE header / metadata model will be our default one
    # but eventually we'll have multiple classifiers built using the HANN model
    hann = HierarchicalAttentionNetwork()
    # hann.feature_keys = feature_keys
    print("Training file {}.".format(training_fp))
    model = hann.read_and_train_data(training_fp, model_base=model_base)
    print("Training successful.")
    if model_save:
        hann.save_model(model, model_fp)
        print("Saving model to {}.".format(model_fp))
    return (model, (hann.X, hann.Y))


if __name__ == "__main__":
    training_fp = DEFAULT_TRAINING_DATA_PATH
    # feature_keys = pe_headers_feature_keys
    # feature_keys_list = [dict["name"] for dict in pe_headers_feature_keys]
    feature_keys_list = list(pd.read_csv(training_fp).columns)
    # Classification label can't be considered a feature (for training the model
    # at least), so we'll filter that out
    feature_keys_filtered = [
        key for key in feature_keys_list if key is not "classification" or "class"
    ]
    # feature_keys_list.append("classification")
    target_feature = "classification"
    # Get our most important features from the training data
    # Using Pearson correlation is appropriate because we only have
    # two categories (point-biserial correlation).
    # If we had multi-class dataset there'd need to be
    # additional preprocessing done.
    corr = get_features_corr(training_fp, feature_keys_list)[0]["corr"]
    stats_vis.visualize_correlation_matrix(
        corr,
        annot=False,
        plot_title="Features Correlation for PE Header Data",
        save_image=True,
        show=True,
    )
    # Now out of those, let's get the top N features
    stats_vis.visualize_correlation_matrix(
        corr,
        annot=True,
        plot_title="Features Correlation for PE Header Data",
        save_image=True,
        show=True,
    )
    results = get_features_corr(training_fp, feature_keys_list, target_feature)
    # Let's make a LogisticRegression model first, to use as a baseline comparison
    model_base = LogisticRegression(random_state=1618)
    model_results = train_model(training_fp, feature_keys_filtered, model_base=model_base)
    model = model_results[0]
