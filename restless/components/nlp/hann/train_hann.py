import os, sys
import pandas as pd

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
    print("DF COLUMNS: ", df.columns)
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
            print("\tCorrelation for {} is {}.".format(feature, corr))
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


def train_hann_model(
    training_fp: str,
    feature_keys: dict,
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
        model_save (bool, optional): Whether to save trained model to disk.
        model_fp (str, optional): Filepath to save the model to, if
            model_save is set to True. Will default if not specified.

    Returns:
        object: Trained model.
    """
    # For now the PE header / metadata model will be our default one
    # but eventually we'll have multiple classifiers built using the HANN model
    hann = HierarchicalAttentionNetwork()
    hann.feature_keys = feature_keys
    print(
        "Training file {} with feature keys: {}.".format(training_fp, hann.feature_keys)
    )
    model = hann.read_and_train_data(training_fp)
    print("Training successful.")
    if model_save:
        hann.save_model(model, model_fp)
        print("Saving model to {}.".format(model_fp))
    return model


if __name__ == "__main__":
    training_fp = DEFAULT_TRAINING_DATA_PATH
    feature_keys = pe_headers_feature_keys
    feature_keys_list = [dict["name"] for dict in pe_headers_feature_keys]
    feature_keys_list.append("classification")
    target_feature = "classification"
    # Let's see our most important features from the training data
    corr = get_features_corr(training_fp, feature_keys_list)[0]["corr"]
    stats_vis.visualize_correlation_diagonal_matrix(
        corr,
        plot_title="Features Correlation for PE Header Data",
        save_image=True,
        show=True,
    )
    stats_vis.visualize_correlation(
        corr,
        plot_title="Features Correlation for PE Header Data",
        save_image=True,
        show=True,
    )
    results = get_features_corr(training_fp, feature_keys_list, target_feature)
    train_hann_model(training_fp, feature_keys)
