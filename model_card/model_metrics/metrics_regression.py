import numpy as np
import pandas as pd

from timeit import default_timer as timer
from loguru import logger
from operator import itemgetter
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    max_error,
    confusion_matrix,
)

from sklearn.preprocessing import KBinsDiscretizer


class RegressionModelAssessment():
    def __init__(
        self,
        model,
        training_dataset: pd.DataFrame,
        validation_dataset: pd.DataFrame,
        kmeans_centroids: np.ndarray,
    ):
        self.discretizer = KBinsDiscretizer(
            n_bins=5, encode="ordinal", strategy="kmeans"
        )
        self.discretizer.fit(training_dataset.get_y().to_numpy().reshape(-1, 1))

        self.discretizer_edges = {
            0: "< {}".format(round(float(self.discretizer.bin_edges_[0][1]), 2)),
            1: ">= {}, < {}".format(
                round(float(self.discretizer.bin_edges_[0][1]), 2),
                round(float(self.discretizer.bin_edges_[0][2]), 2),
            ),
            2: ">= {}, < {}".format(
                round(float(self.discretizer.bin_edges_[0][2]), 2),
                round(float(self.discretizer.bin_edges_[0][3]), 2),
            ),
            3: ">= {}, < {}".format(
                round(float(self.discretizer.bin_edges_[0][3]), 2),
                round(float(self.discretizer.bin_edges_[0][4]), 2),
            ),
            4: "> {}".format(round(float(self.discretizer.bin_edges_[0][4]), 2)),
        }

        ModelAssessmentInterface.__init__(
            self, model, training_dataset, validation_dataset, VAE, kmeans_centroids
        )

    def get_prediction(
        self, instance: pd.DataFrame, preprocess: bool = True, prepare_data: bool = True
    ):
        try:
            predictions = self.model.predict(instance, preprocess, prepare_data)
        except Exception:
            raise Exception(
                "Error with the loaded model, check that it has been trained with the same dataset uploaded."
            )
        if len(predictions.shape) < 2:
            predictions = predictions.reshape(-1, 1)
        predicted_bin = self.discretizer.transform(predictions)

        return predictions, predicted_bin

    def normalize_y(self, y):
        y_mean = self.training_dataset.get_y_mean()
        y_std = self.training_dataset.get_y_std()
        return (y - y_mean) / y_std

    def get_sanity_check(self):
        y_training_predicted, _ = self.get_prediction(self.training_dataset.get_x())

        y_test_predicted, _ = self.get_prediction(self.validation_dataset.get_x())
        y_training_true = self.training_dataset.get_y()
        y_test_true = self.validation_dataset.get_y()

        y_predicted = np.concatenate((y_training_predicted, y_test_predicted))
        y_true = np.concatenate((y_training_true, y_test_true))

        min_true = (round(float(np.min(y_true)), 4),)
        max_true = (round(float(np.max(y_true)), 4),)
        min_predicted = (round(float(np.min(y_predicted)), 4),)
        max_predicted = (round(float(np.max(y_predicted)), 4),)
        len_predicted_uniques = len(np.unique(y_predicted))

        sanity_check = {
            "min_true": min_true,
            "max_true": max_true,
            "min_predicted": min_predicted,
            "max_predicted": max_predicted,
            "min_outbound": min_predicted < min_true,
            "max_outbound": max_predicted > max_true,
            "len_predicted_uniques": len_predicted_uniques,
        }

        return sanity_check

    def get_model_metrics(self, dataset: Dataset):
        y_true = dataset.get_y()
        y_predicted, _ = self.get_prediction(dataset.get_x())

        metrics = {
            "mse": round(float(mean_squared_error(y_true, y_predicted)), 4),
            "rmse": round(
                float(mean_squared_error(y_true, y_predicted, squared=False)), 4
            ),
            "mae": round(float(mean_absolute_error(y_true, y_predicted)), 4),
            "max_error": round(float(max_error(y_true, y_predicted)), 4),
            "r2_score": round(float(r2_score(y_true, y_predicted)), 4),
        }

        return metrics

    def get_descriptive_curve(self, dataset: Dataset):
        idx = np.random.choice(len(dataset.get_x()), size=120)

        y_true = dataset.get_y().to_numpy()[idx].reshape(1, -1)[0]
        y_predicted = self.get_prediction(dataset.get_x().iloc[idx])[0].reshape(1, -1)[
            0
        ]

        mse = [
            mean_squared_error(y_true[i : i + 1], y_predicted[i : i + 1])
            for i, _ in enumerate(y_true)
        ]

        curves = {
            "scatter_plot": {
                "true": [round(float(value), 4) for value in y_true],
                "predicted": [round(float(value), 4) for value in y_predicted],
            },
            "mse_plot": {
                "mse": [round(float(value), 4) for value in mse],
                "predicted": [round(float(value), 4) for value in y_predicted],
            },
        }

        return curves

    def get_error_matrix(self):
        start = timer()

        y_test_true = self.validation_dataset.get_y()
        y_test_true_bin = self.discretizer.transform(
            y_test_true.to_numpy().reshape(-1, 1)
        )
        _, y_test_predicted_bin = self.get_prediction(self.validation_dataset.get_x())

        y_test_true_bin_label = [
            self.discretizer_edges[i] for i in y_test_true_bin.reshape(1, -1)[0]
        ]
        y_test_predicted_bin_label = [
            self.discretizer_edges[i] for i in y_test_predicted_bin.reshape(1, -1)[0]
        ]

        error_matrix_json = []

        for i, row in enumerate(
            confusion_matrix(
                y_test_true_bin_label,
                y_test_predicted_bin_label,
                labels=list(self.discretizer_edges.values()),
            )
        ):
            for j, element in enumerate(row):
                examples = []
                subgroup = np.where(
                    (y_test_true_bin == i) & (y_test_predicted_bin == j)
                )[0]
                if len(subgroup) == 0:
                    examples.append(
                        {
                            "instance": [],
                            "lime_values": [],
                            "lime_labels": [],
                            "decision_rule": [],
                            "prediction": {},
                        }
                    )
                elif len(subgroup) == 1:
                    prediction = self.get_prediction(
                        self.validation_dataset.get_x().iloc[
                            subgroup[0] : subgroup[0] + 1
                        ]
                    )[0][0][0]
                    true_value = self.validation_dataset.get_y().iloc[subgroup[0]]
                    decision_path, score, LIME_explanation = self.get_explanation(
                        self.validation_dataset.get_x().iloc[
                            subgroup[0] : subgroup[0] + 1
                        ]
                    )
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
                            "decision_rule": decision_path,
                            "explanation_score": score,
                            "LIME": LIME_explanation,
                            "prediction": {
                                "true_value": float(true_value),
                                "predicted_value": round(float(prediction), 4),
                            },
                        }
                    )
                else:
                    instances = self._find_representative_examples(subgroup)
                    for index in instances:
                        prediction = self.get_prediction(
                            self.validation_dataset.get_x().iloc[index : index + 1]
                        )[0][0][0]
                        true_value = self.validation_dataset.get_y().iloc[index]
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
                                "prediction": {
                                    "true_value": float(true_value),
                                    "predicted_value": round(float(prediction), 4),
                                },
                            }
                        )
                error_matrix_json.append(
                    {
                        "true_label": str(self.discretizer_edges[i]),
                        "predicted_label": str(self.discretizer_edges[j]),
                        "n_cases": int(element),
                        "examples": examples,
                    }
                )

        end = timer()
        logger.debug("Error matrix computed in {:.3f}s".format((end - start)))

        return error_matrix_json

    def get_model_limitations(self):
        start = timer()

        y_test_true = self.validation_dataset.get_y()
        y_test_true_bin = self.discretizer.transform(
            y_test_true.to_numpy().reshape(-1, 1)
        )
        y_test_predicted, y_test_predicted_bin = self.get_prediction(
            self.validation_dataset.get_x()
        )

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

            if validation_cluster_instances.shape[0] > 0:
                mse = mean_squared_error(
                    y_test_true[validation_cluster_instances],
                    y_test_predicted[validation_cluster_instances],
                )
                mae = mean_absolute_error(
                    y_test_true[validation_cluster_instances],
                    y_test_predicted[validation_cluster_instances],
                )
            else:
                mse = 0
                mae = 0
            metrics["mse"] = round(float(mse), 4) if not np.isnan(mse) else float(-1)
            metrics["mae"] = round(float(mae), 4) if not np.isnan(mae) else float(-1)

            metrics["labels_distribution"] = {}
            for predicted_bin, bin_edges in sorted(self.discretizer_edges.items()):
                metrics["labels_distribution"][str(bin_edges)] = {
                    "true_labels": int(
                        np.where(
                            y_test_true_bin[validation_cluster_instances]
                            == predicted_bin
                        )[0].shape[0]
                    ),
                    "predicted": int(
                        np.where(
                            y_test_predicted_bin[validation_cluster_instances]
                            == predicted_bin
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
                                features[column]["NaN"] = float(
                                    round(
                                        cluster_df[column].isna().sum()
                                        / n_training_instances,
                                        4,
                                    )
                                )
                            elif isinstance(value, float):
                                features[column][value] = float(
                                    round(
                                        cluster_df[column].isna().sum()
                                        / n_training_instances,
                                        4,
                                    )
                                )
                            else:
                                features[column][value] = float(
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

        end = timer()
        logger.debug("Model limitations found in {:.3f}s".format((end - start)))

        return sorted(cluster_metrics, key=itemgetter("mse"), reverse=True)