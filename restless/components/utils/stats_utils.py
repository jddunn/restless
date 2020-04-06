import os
import collections, functools, operator

from pandas import DataFrame

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    log_loss,
)
from sklearn.model_selection import cross_val_score

# Path to save visualization output images
DEFAULT_SCREENSHOTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "screenshots")
)


class StatsUtils:
    """
    Contains functions to facilitate statistical modeling
    and ML processing.
    """

    def __init__(self):
        pass

    def get_correlation_for_features(
        self,
        data_train: DataFrame,
        features_to_compare: list = [],
        target_feature: str = None,
        correlation_method: str = "pearson",
        print_output: bool = False,
    ):
        """
        Gets correlation for a feature with all other features
        in a dataframe, if $features param is not provided.

        Args:
            data_train (DataFrame): Pandas df for data
            target_feature (str, optional): Name of feature (column)
                to get correlation values with (e.g. classification)
            features_to_compare (list, optional): Optional; if set,
                function will get the correlation values for
                just the specified features.
            correlation_method (str, optional): Type of correlation
                metric to use; defaults to "pearson."
            print_output (bool, optional): Whether to output of results.
        """
        features_to_compare.append(target_feature)
        data_train = data_train.filter(features_to_compare)
        corr = data_train.corr(method=correlation_method)
        if print_output:
            print(
                "Correlation for features: {} is {}".format(features_to_compare, corr)
            )
        return corr

    def get_model_metrics(
        self, y, y_pred, labels: list = ["0", "1"], print_output: bool = False,
    ) -> dict:
        """
        Gets metrics from evaluated model and input data, including
        accuracy, loss, confusion matrix (if applicable), F1 score,
        etc.
        """
        result = {}
        cm = None
        _labels = [0, 1]
        try:
            # we can only get cms from binary classifiers
            cm = confusion_matrix(y, y_pred, _labels)
        except:
            pass
        result["cm"] = cm
        accuracy = accuracy_score(y_pred, y)
        result["accuracy"] = accuracy
        loss = log_loss(y, y_pred)
        result["loss"] = loss
        precision = precision_score(y_pred, y)
        result["precision"] = precision
        recall = recall_score(y_pred, y)
        result["recall"] = recall
        f1 = f1_score(y_pred, y)
        result["f1"] = f1
        # cross_val = cross_val_score(y_pred, y)
        # result["cross_val"] = cross_val
        kappa = cohen_kappa_score(y_pred, y)
        result["kappa"] = kappa
        auc = roc_auc_score(y_pred, y)
        result["auc"] = auc
        if print_output:
            print("Model evaluation metrics: ")
            print("\tConfusion matrix: ", cm)
            try:
                print("\t", self.pretty_print_cm(cm, labels))
            except Exception as e:
                print(e)
                pass
            print("\tAccuracy: {} \tLoss: {}".format(accuracy, loss))
            print("\tPrecision: {}".format(precision))
            print("\tRecall: {}".format(recall))
            print("\tF1 score: {}".format(f1))
            # print("\tCross-val score: {}".format(cross_val))
            print("\tCohens kappa score: {}".format(kappa))
            print("\tROC AUC score: {}".format(auc))
        return result

    def get_metrics_averages(self, metrics: list) -> dict:
        """
        Takes a list of dictionaries and returns their averaged
        values for each key in each list. Values in the dictionary
        must be int or floats.

        Args:
            dicts (list): A list of dictionaries to to average.

        Returns:
            dict: Dictionary with averaged values.
        """
        result = dict(functools.reduce(operator.add, map(collections.Counter, metrics)))
        result.update({n: result[n] / len(metrics) for n in result.keys()})
        return result

    def pretty_print_cm(
        self,
        cm,
        labels: list,
        hide_zeroes: bool = False,
        hide_diagonal: bool = False,
        hide_threshold: bool = None,
    ) -> None:
        """
        Pretty print for confusion matrixes.
        https://gist.github.com/zachguo/10296432
        """
        # Print header
        columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
        empty_cell = " " * columnwidth
        # Print header
        print("    " + empty_cell, end=" ")
        for label in labels:
            print("%{0}s".format(columnwidth) % label, end=" ")
        print()
        # Print rows
        for i, label1 in enumerate(labels):
            print("    %{0}s".format(columnwidth) % label1, end=" ")
            for j in range(len(labels)):
                cell = "%{0}.1f".format(columnwidth) % cm[i, j]
                if hide_zeroes:
                    cell = cell if float(cm[i, j]) != 0 else empty_cell
                if hide_diagonal:
                    cell = cell if i != j else empty_cell
                if hide_threshold:
                    cell = cell if cm[i, j] > hide_threshold else empty_cell
                print(cell, end=" ")
            print()
        return
