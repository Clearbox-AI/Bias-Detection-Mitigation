import numpy as np
import pandas as pd

from operator import itemgetter
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


class ClassificationModelAssessment():

    def init(self, x):
        self.training_dataset = x
        self.get_prediction

    def get_sanity_check(self):
        y_training_probabilities, y_training_predicted = self.get_prediction(
            self.training_dataset.get_x()
        )
        y_test_probabilities, y_test_predicted = self.get_prediction(
            self.validation_dataset.get_x()
        )

        y_training_true = self.training_dataset.get_y()
        y_test_true = self.validation_dataset.get_y()

        _, encoding = self.training_dataset.get_label_encoded_y()

        y_training_predicted = [encoding[value] for value in y_training_predicted]
        y_test_predicted = [encoding[value] for value in y_test_predicted]

        y_probabilities = np.concatenate(
            (y_training_probabilities, y_test_probabilities)
        )
        y_predicted = np.concatenate((y_training_predicted, y_test_predicted))
        y_true = np.concatenate((y_training_true, y_test_true))

        probabilities_outbound = False
        for probabilities in y_probabilities:
            if (
                probabilities[0] < 0
                or probabilities[0] > 1
                or probabilities[1] < 0
                or probabilities[1] > 1
            ):
                probabilities_outbound = True
                break

        true_uniques = np.unique(y_true)
        predicted_uniques = np.unique(y_predicted)
        classes_not_predicted = np.setdiff1d(true_uniques, predicted_uniques)

        sanity_check = {
            "probabilities_outbound": probabilities_outbound,
            "classes_not_predicted": [
                str(not_predicted) for not_predicted in classes_not_predicted
            ],
        }

        return sanity_check

    def get_model_metrics(self):
        y_test, y_labels = dataset.get_label_encoded_y()
        y_test_binarized = dataset.get_one_hot_encoded_y()
        y_test_predicted_proba, y_test_predicted = self.get_prediction(dataset.get_x())
        precisions, recalls, fscores, supports = precision_recall_fscore_support(
            y_test, y_test_predicted
        )

        class_metrics = []
        for label in np.unique(y_test):
            avg_precision = average_precision_score(
                y_test_binarized[:, label], y_test_predicted_proba[:, label]
            )
            class_metrics.append(
                {
                    "label": str(y_labels[label]),
                    "precision": round(float(precisions[label]), 4),
                    "recall": round(float(recalls[label]), 4),
                    "f1": round(float(fscores[label]), 4),
                    "support": round(float(supports[label]), 4),
                    "average_precision": round(float(avg_precision), 4),
                }
            )
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_test_predicted)), 5),
            "classes": class_metrics,
        }

        return metrics

    def get_descriptive_curve(self):
        y_test, y_labels = dataset.get_label_encoded_y()
        y_test_binarized = dataset.get_one_hot_encoded_y()
        y_test_predicted_proba, _ = self.get_prediction(dataset.get_x())

        calib_curve = []
        precision_and_recall_curve = []

        for label in np.unique(y_test):
            fop, mpv = calibration_curve(
                y_test_binarized[:, label], y_test_predicted_proba[:, label], n_bins=10
            )
            calib_curve.append(
                {"label": str(y_labels[label]), "x": mpv.tolist(), "y": fop.tolist()}
            )

            precision_curve, recall_curve, _ = precision_recall_curve(
                y_test_binarized[:, label],
                np.round(y_test_predicted_proba[:, label], 2),
            )

            avg_precision = average_precision_score(
                y_test_binarized[:, label], y_test_predicted_proba[:, label]
            )

            precision_and_recall_curve.append(
                {
                    "label": str(y_labels[label]),
                    "precision": [round(float(value), 4) for value in precision_curve],
                    "recall": [round(float(value), 4) for value in recall_curve],
                    "average_precision": round(float(avg_precision), 4),
                }
            )

        curves = {
            "calibration_curve": calib_curve,
            "precision_and_recall_curve": precision_and_recall_curve,
        }

        return curves

    def get_error_matrix(self):

        y_true = self.validation_dataset.get_y()

        _, encoding = self.training_dataset.get_label_encoded_y()

        _, y_pred = self.get_prediction(self.validation_dataset.get_x())
        y_pred = [encoding[value] for value in y_pred]

        error_matrix_json = []

        for i, row in enumerate(confusion_matrix(y_true, y_pred, labels=encoding)):
            for j, element in enumerate(row):
                examples = []
                subgroup = np.where(
                    (y_true == encoding[i]) & (np.asarray(y_pred) == encoding[j])
                )[0]
                if len(subgroup) == 0:
                    examples.append(
                        {
                            "instance": [],
                            "probas": {},
                        }
                    )
                elif len(subgroup) == 1:
                    probas = {}
                    prediction = self.get_prediction(
                        self.validation_dataset.get_x().iloc[
                            subgroup[0] : subgroup[0] + 1
                        ]
                    )[0][0]
                    for x, label in enumerate(encoding):
                        probas[str(label)] = round(float(prediction[x]), 4)

                    examples.append(
                        {
                            "instance": dict(
                                zip(
                                    self.validation_dataset.x_columns(),
                                    [
                                        "NaN" if pd.isnull(value) else str(value)
                                        for value in self.validation_dataset.get_x()
                                        .iloc[subgroup[0]]
                                        .values.tolist()
                                    ],
                                )
                            ),
                            "probas": probas,
                        }
                    )
                else:
                    instances = self._find_representative_examples(subgroup)
                    for index in instances:
                        probas = {}
                        prediction = self.get_prediction(
                            self.validation_dataset.get_x().iloc[index : index + 1]
                        )[0][0]
                        for x, label in enumerate(encoding):
                            probas[str(label)] = round(float(prediction[x]), 4)
                        decision_path, score, LIME_explanation = self.get_explanation(
                            self.validation_dataset.get_x().iloc[index : index + 1]
                        )
                        examples.append(
                            {
                                "instance": dict(
                                    zip(
                                        self.validation_dataset.x_columns(),
                                        [
                                            "NaN" if pd.isnull(value) else str(value)
                                            for value in self.validation_dataset.get_x()
                                            .iloc[index]
                                            .values.tolist()
                                        ],
                                    )
                                ),
                                "decision_rule": decision_path,
                                "explanation_score": score,
                                "LIME": LIME_explanation,
                                "probas": probas,
                            }
                        )
                error_matrix_json.append(
                    {
                        "true_label": str(encoding[i]),
                        "predicted_label": str(encoding[j]),
                        "n_cases": int(element),
                        "examples": examples,
                    }
                )

        end = timer()
        logger.debug("Error matrix computed in {:.3f}s".format((end - start)))

        return error_matrix_json

    def get_model_limitations(self):
        start = timer()
        _, y_test_predicted = self.get_prediction(self.validation_dataset.get_x())
        y_test, true_classes = self.validation_dataset.get_label_encoded_y()

        cluster_metrics = []

        for cluster in np.unique(self.training_cluster_labels):
            training_cluster_instances = np.where(
                self.training_cluster_labels == cluster
            )[0]
            validation_cluster_instances = np.where(
                self.validation_cluster_labels == cluster
            )[0]

            n_training_instances = training_cluster_instances.shape[0]
            n_validation_instances = validation_cluster_instances.shape[0]

            metrics = {}
            metrics["validation_instances"] = n_validation_instances
            metrics["instances"] = int(training_cluster_instances.shape[0])

            accuracy = accuracy_score(
                y_test[validation_cluster_instances],
                y_test_predicted[validation_cluster_instances],
            )
            metrics["accuracy"] = (
                round(float(accuracy), 4) if not np.isnan(accuracy) else float(-1)
            )

            metrics["labels_distribution"] = {}
            for encoded_label in np.unique(y_test):
                metrics["labels_distribution"][str(true_classes[encoded_label])] = {
                    "true_labels": int(
                        np.where(y_test[validation_cluster_instances] == encoded_label)[
                            0
                        ].shape[0]
                    ),
                    "predicted": int(
                        np.where(
                            y_test_predicted[validation_cluster_instances]
                            == encoded_label
                        )[0].shape[0]
                    ),
                }

            cluster_df = self.training_dataset.get_x().iloc[training_cluster_instances]

            features = {}
            for column in cluster_df.columns:
                if column in self.preprocessor.get_ordinal_features():
                    features[str(column)] = {}
                    features[str(column)]["mean"] = (
                        "NaN"
                        if pd.isnull(float(cluster_df[column].mean()))
                        else round(float(cluster_df[column].mean()), 4)
                    )
                    features[str(column)]["median"] = (
                        "NaN"
                        if pd.isnull(float(cluster_df[column].median()))
                        else round(float(cluster_df[column].median()), 4)
                    )
                    features[str(column)]["q1"] = (
                        "NaN"
                        if pd.isnull(float(cluster_df[column].quantile(0.25)))
                        else round(float(cluster_df[column].quantile(0.25)), 4)
                    )
                    features[str(column)]["q3"] = (
                        "NaN"
                        if pd.isnull(float(cluster_df[column].quantile(0.75)))
                        else round(float(cluster_df[column].quantile(0.75)), 4)
                    )
                    features[str(column)]["std"] = (
                        "NaN"
                        if pd.isnull(float(cluster_df[column].std()))
                        else round(float(cluster_df[column].std()), 4)
                    )
                    features[str(column)]["min"] = (
                        "NaN"
                        if pd.isnull(float(cluster_df[column].min()))
                        else round(float(cluster_df[column].min()), 4)
                    )
                    features[str(column)]["max"] = (
                        "NaN"
                        if pd.isnull(float(cluster_df[column].max()))
                        else round(float(cluster_df[column].max()), 4)
                    )
                if column in self.preprocessor.get_categorical_features():
                    if len(cluster_df[column].unique().tolist()) < (
                        0.8 * len(cluster_df)
                    ):
                        features[str(column)] = {}
                        for value in cluster_df[column].unique().tolist():
                            if pd.isnull(value):
                                features[str(column)]["NaN"] = float(
                                    round(
                                        cluster_df[column].isna().sum()
                                        / n_training_instances,
                                        4,
                                    )
                                )
                            elif isinstance(value, float):
                                features[str(column)][str(value)] = float(
                                    round(
                                        cluster_df[column].isna().sum()
                                        / n_training_instances,
                                        4,
                                    )
                                )
                            else:
                                features[str(column)][str(value)] = float(
                                    round(
                                        np.where(cluster_df[column] == value)[0].shape[
                                            0
                                        ]
                                        / n_training_instances,
                                        4,
                                    )
                                )
                if column in self.preprocessor.get_datetime_features():
                    features[str(column)] = {}
                    features[str(column)]["unique_values"] = (
                        "NaN"
                        if pd.isnull(float(cluster_df[column].nunique()))
                        else round(float(cluster_df[column].nunique()), 4),
                    )
                    features[str(column)]["min"] = (
                        "NaN"
                        if pd.isnull(cluster_df[column].min())
                        else cluster_df[column].min()
                    )
                    features[str(column)]["max"] = (
                        "NaN"
                        if pd.isnull(cluster_df[column].max())
                        else cluster_df[column].max()
                    )
                    features[str(column)]["most_frequent"] = (
                        "NaN"
                        if pd.isnull(cluster_df[column].value_counts().idxmax())
                        else cluster_df[column].value_counts().idxmax()
                    )

            metrics["features"] = features

            cluster_metrics.append(metrics)

        return