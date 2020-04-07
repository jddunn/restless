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
    from restless.components.utils import utils
except:
    from hann import HierarchicalAttentionNetwork
    from hann import (
        DEFAULT_DATA_PATH,
        DEFAULT_TRAINING_DATA_PATH,
        DEFAULT_MODEL_PATH,
    )
    from ...utils import utils

stats = utils.stats
stats_vis = utils.stats_vis
scaler = RobustScaler()

MAX_N_FEATURES = None  # If none we have no limit
CORR_THRESHOLD = 0.1  # minimum val to consider meaningful linear correlation


def get_features_corr(
    df: pd.DataFrame,
    features: list,
    target_feature: str = None,
    get_corr_with_target_feature_only: bool = False,
    correlation="pearson",
) -> list:
    """
    Gets correlation of each feature in a list compared to the classification
    probability of the model.

    Args:
        df (pd.DataFrame): DataFrrame with features (columns) to process.
        features (list): List of features to get correlations with
        target_feature (str, optional): If specified, will return correlation
            values for features with target_feature only (if
            $get_corr_with_target_feature_only is True).
        get_corr_with_target_feature_only (bool, optional): If specified,
            then only correlations with the $target_feature will be returned.
        correlation (str, optional): Type of correlation metric to classify;
            defaults to "jaccard".

    Returns:
        list: List of correlations; if no $target_feature param is specified, the
            list will only contain one result, with all the feature correlations.
    """
    results = []
    df = pd.read_csv(training_fp)
    if target_feature:
        result = {}
        corr = stats.get_correlation_for_features(
            df,
            features,
            target_feature=target_feature,
            get_corr_with_target_feature_only=get_corr_with_target_feature_only,
            correlation=correlation,
        )
        print(
            "Getting correlation for features {} with target feature {}.".format(
                features, target_feature
            )
        )
        result["features"] = features
        result["target_feature"] = target_feature
        result["corr"] = corr
        print("\t", corr)
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
    top_features: list = None,
    labels: list = None,
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
        top_features (list, optional): List of top features (all other features will
            be dropped from df to train); optional.
        labels (list, optional): List of labels (used for labelling charts
            and model metrics).
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
    model = hann.read_and_train_data(
        training_fp, top_features=top_features, model_base=model_base, labels=labels
    )
    print("Training successful.")
    if model_save:
        hann.save_model(model, model_fp)
        print("Saving model to {}.".format(model_fp))
    return (model, (hann.X, hann.Y))


if __name__ == "__main__":
    labels = ["benign", "malicious"]
    training_fp = DEFAULT_TRAINING_DATA_PATH
    df = pd.read_csv(training_fp)
    feature_keys_list = list(df.columns)
    # Classification label can't be considered a feature (for training the model
    # at least), so we'll filter that out
    feature_keys_filtered = [
        key for key in feature_keys_list if key != "classification"
    ]
    target_feature = "classification"
    # Get our most important features from the training data
    # Using Pearson correlation is appropriate because we only have
    # two categories (point-biserial correlation).
    corr = get_features_corr(df, feature_keys_list, target_feature=target_feature)[0][
        "corr"
    ]
    stats_vis.visualize_correlation_matrix(
        corr,
        feature_keys_list,
        annot=False,
        plot_title="Features Correlation Matrix for PE Header Data",
        save_image=True,
        show=True,
    )
    # Now out of those, let's get the top N features
    target_feature = "classification"
    print(
        "Transforming df into df with top extracted features from list: {}.".format(
            feature_keys_filtered
        )
    )
    target_corr = get_features_corr(
        df,
        feature_keys_list,
        target_feature=target_feature,
        get_corr_with_target_feature_only=True,
    )[0]["corr"]
    top_df, top_features = stats.transform_df_with_top_features_for_hann(
        df, target_corr, feature_keys_list, target_feature, threshold=CORR_THRESHOLD
    )
    _top_features = top_features
    _top_features.append("classification")  # For our labels
    top_corr = get_features_corr(
        top_df,
        _top_features,
        target_feature=target_feature,
        get_corr_with_target_feature_only=False,
    )[0]["corr"]
    stats_vis.visualize_correlation_matrix(
        top_corr,
        _top_features,
        annot=True,
        plot_title="Top Features Correlation Matrix for PE Header Data (Minimum threshold of "
        + str(CORR_THRESHOLD)
        + ")",
        save_image=True,
        show=True,
    )
    # Let's make a LogisticRegression model first, to use as a baseline comparison
    # model_base = LogisticRegression(random_state=1618)
    # If we don't pass a model_base, will train HANN architecture by default.
    model_base = None
    model_results = train_model(
        training_fp,
        feature_keys=feature_keys_filtered,
        top_features=top_features,
        labels=labels,
        model_base=model_base,
    )
    model = model_results[0]
