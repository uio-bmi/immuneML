from pathlib import Path

import logging
import pandas as pd
import os
import warnings

import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import lognorm

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.dsl.instruction_parsers.LabelHelper import LabelHelper
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.encodings.motif_encoding.PositionalMotifHelper import PositionalMotifHelper
from immuneML.ml_methods.util.Util import Util
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.steps.DataEncoder import DataEncoder
from immuneML.workflows.steps.DataEncoderParams import DataEncoderParams


class MotifGeneralizationAnalysis(DataReport):
    """
    This report splits the given dataset into a training and test set, identifies significant motifs using the
    :py:obj:`~immuneML.encodings.motif_encoding.MotifEncoder.MotifEncoder`
    on the training set and plots the precision/recall and precision/true positive predictions of motifs
    on both the training and test sets. This can be used to investigate generalizability of motifs and calibrate
    example weighting parameters.

    Arguments:

        label (dict): A label configuration. One label should be specified, and the positive_class for this label should be defined. See the YAML specification below for an example.

        highlight_motifs_path (str): path to a set of motifs of interest to highlight in the output figures. By default no motifs are highlighted.

        highlight_motifs_name (str): if highlight_motifs_path is defined, this name will be used to label the motifs of interest in the output figures.

        smoothen_combined_precision (bool): whether to add a smoothed line representing the combined precision to the precision-vs-TP plot. When set to True, this may take considerable extra time to compute. By default, plot_smoothed_combined_precision is set to True.

        training_set_identifier_path (str): path to a file containing 'sequence_identifiers' of the sequences used for the training set. Each line in the file should represent one sequence identifier, with no file header. The remaining sequences will be used as the validation set. If training_set_identifier_path is not set, a random subset of the data (according to training_percentage) will be assigned to be the training set.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml


        my_report:
            MotifGeneralizationAnalysis:
                ...
                label: # Define a label, and the positive class for that given label
                    CMV:
                        positive_class: +


    """

    def __init__(self, training_set_identifier_path: str = None, training_percentage: float = None,
                 max_positions: int = None, min_precision: float = None,
                 min_recall: float = None, min_true_positives: int = None,  random_seed: int = None, label: dict = None,
                 smoothen_combined_precision: bool = None,
                 min_points_in_window: int = None, smoothing_constant1: float = None, smoothing_constant2: float = None,
                 highlight_motifs_path: str = None, highlight_motifs_name: str = None,
                 dataset: SequenceDataset = None, result_path: Path = None, number_of_processes: int = 1, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.training_set_identifier_path = Path(training_set_identifier_path) if training_set_identifier_path is not None else None
        self.training_percentage = training_percentage
        self.max_positions = max_positions
        self.max_positions = max_positions
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.min_true_positives = min_true_positives
        self.smoothen_combined_precision = smoothen_combined_precision
        self.min_points_in_window = min_points_in_window
        self.smoothing_constant1 = smoothing_constant1
        self.smoothing_constant2 = smoothing_constant2
        self.random_seed = random_seed
        self.label = label
        self.col_names = None
        self.tp_cutoff = None
        self.recall_cutoff = None

        self.highlight_motifs_name = highlight_motifs_name
        self.highlight_motifs_path = Path(highlight_motifs_path) if highlight_motifs_path is not None else None

    @classmethod
    def build_object(cls, **kwargs):
        location = MotifGeneralizationAnalysis.__name__

        ParameterValidator.assert_type_and_value(kwargs["max_positions"], int, location, "max_positions", min_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["min_precision"], (int, float), location, "min_precision", min_inclusive=0, max_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["min_recall"], (int, float), location, "min_recall", min_inclusive=0, max_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["min_true_positives"], int, location, "min_true_positives", min_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["smoothen_combined_precision"], bool, location, "smoothen_combined_precision")
        ParameterValidator.assert_type_and_value(kwargs["min_points_in_window"], int, location, "min_points_in_window", min_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["smoothing_constant1"], (int, float), location, "smoothing_constant1", min_exclusive=0)
        ParameterValidator.assert_type_and_value(kwargs["smoothing_constant2"], (int, float), location, "smoothing_constant2", min_exclusive=0)

        if kwargs["training_set_identifier_path"] is not None:
            ParameterValidator.assert_type_and_value(kwargs["training_set_identifier_path"], str, location, "training_set_identifier_path")
            assert os.path.isfile(kwargs["training_set_identifier_path"]), f"{location}: the file {kwargs['training_set_identifier_path']} does not exist. " \
                                         f"Specify the correct path under training_set_identifier_path."
        else:
            ParameterValidator.assert_type_and_value(kwargs["training_percentage"], float, location, "training_percentage", min_exclusive=0, max_exclusive=1)

        if "random_seed" in kwargs and kwargs["random_seed"] is not None:
            ParameterValidator.assert_type_and_value(kwargs["random_seed"], int, location, "random_seed")

        ParameterValidator.assert_type_and_value(kwargs["label"], (dict, str), location, "label")
        if type(kwargs["label"]) is dict:
           assert len(kwargs["label"]) == 1, f"{location}: The number of specified labels must be 1, found {len(kwargs['label'])}: {', '.join(list(len(kwargs['label'].keys())))}"

        if "highlight_motifs_path" in kwargs and kwargs["highlight_motifs_path"] is not None:
            PositionalMotifHelper.check_motif_filepath(kwargs["highlight_motifs_path"], location, "highlight_motifs_path")

        ParameterValidator.assert_type_and_value(kwargs["highlight_motifs_name"], str, location, "highlight_motifs_name")

        return MotifGeneralizationAnalysis(**kwargs)

    def _generate(self):
        self._set_colnames()

        encoded_training_data, encoded_test_data = self._get_encoded_train_test_data()
        training_plotting_data, test_plotting_data = self._get_plotting_data(encoded_training_data, encoded_test_data)

        training_combined_precision = self._get_combined_precision(training_plotting_data)
        test_combined_precision = self._get_combined_precision(test_plotting_data)
        self._set_tp_cutoff(test_combined_precision)
        self._set_recall_cutoff(encoded_training_data)

        output_tables = self._write_output_tables(training_plotting_data, test_plotting_data, training_combined_precision, test_combined_precision)
        output_text = self._write_stats()

        plots = self._write_plots(training_plotting_data, test_plotting_data, training_combined_precision, test_combined_precision)

        return ReportResult(output_tables=output_tables,
                            output_text=output_text,
                            output_figures=plots)

    def _set_colnames(self):
        self.col_names = dict()

        if self.dataset.get_example_weights() is not None:
            for name in ["precision", "recall"]:
                self.col_names[name] = f"weighted_{name}_scores"

            for name in ["tp", "fp", "fn", "tn"]:
                self.col_names[name] = f"weighted_{name}_count"

            self.col_names["combined precision"] = "Combined weighted precision"
        else:
            for name in ["precision", "recall"]:
                self.col_names[name] = f"{name}_scores"

            for name in ["tp", "fp", "fn", "tn"]:
                self.col_names[name] = f"raw_{name}_count"

            self.col_names["combined precision"] = "Combined precision"

    def _get_encoded_train_test_data(self):
        train_data_path = PathBuilder.build(self.result_path / "datasets/train")
        test_data_path = PathBuilder.build(self.result_path / "datasets/test")

        train_indices, val_indices = self._get_train_val_indices()

        training_data = self.dataset.make_subset(train_indices, train_data_path, Dataset.TRAIN)
        test_data = self.dataset.make_subset(val_indices, test_data_path, Dataset.TEST)

        encoder = self._get_encoder()

        encoded_training_data = self._encode_dataset(training_data, encoder, learn_model=True)
        encoded_test_data = self._encode_dataset(test_data, encoder, learn_model=False)

        return encoded_training_data, encoded_test_data

    def _get_train_val_indices(self):
        if self.training_set_identifier_path is None:
            return Util.get_train_val_indices(self.dataset.get_example_count(),
                                              self.training_percentage, random_seed=self.random_seed)
        else:
            return self._get_train_val_indices_from_file()


    def _get_train_val_indices_from_file(self):
        with open(self.training_set_identifier_path, "r") as file:
            train_identifiers = [identifier.strip() for identifier in file.readlines()]

        train_indices = []
        val_indices = []
        val_identifiers = []

        for idx, sequence in enumerate(self.dataset.get_data()):
            if sequence.identifier in train_identifiers:
                train_indices.append(idx)
            else:
                val_indices.append(idx)
                val_identifiers.append(sequence.identifier)

        logging.info(f"{MotifGeneralizationAnalysis.__name__}: \nTraining set identifiers ({len(train_identifiers)}): {train_identifiers}\nValidation set identifiers ({len(val_identifiers)}): {val_identifiers}")

        assert len(train_indices) > 0, f"{MotifGeneralizationAnalysis.__name__}: error when reading training set identifiers from training_set_identifier_path, 0 of the identifiers were present in the dataset. Please check training_set_identifier_path: {self.training_set_identifier_path}, and see the log file for more information."
        assert len(val_indices) > 0, f"{MotifGeneralizationAnalysis.__name__}: error when inferring validation set identifiers from training_set_identifier_path, all of the identifiers were present in the dataset resulting in 0 sequences in the validation set. Please check training_set_identifier_path: {self.training_set_identifier_path}, and see the log file for more information."
        assert len(train_indices) == len(train_identifiers), f"{MotifGeneralizationAnalysis.__name__}: error when reading training set identifiers from training_set_identifier_path, not all identifiers provided in the file occurred in the dataset ({len(train_indices)} of {len(train_identifiers)} found). Please check training_set_identifier_path: {self.training_set_identifier_path}, and see the log file for more information."

        return train_indices, val_indices

    def _get_encoder(self):
        encoder = MotifEncoder.build_object(self.dataset, **{"max_positions": self.max_positions,
                                                            "min_precision": self.min_precision,
                                                            "min_recall": self.min_recall,
                                                            "min_true_positives": self.min_true_positives,
                                                            "generalize_motifs": False,
                                                            "label": None,
                                                            "name": f"motif_encoder"})

        return encoder

    def _encode_dataset(self, dataset, encoder, learn_model):
        encoded_dataset = DataEncoder.run(DataEncoderParams(dataset=dataset, encoder=encoder,
                                                            encoder_params=EncoderParams(result_path=self.result_path / f"encoded_data/{dataset.name}",
                                                                                         label_config=self._get_label_config(dataset),
                                                                                         pool_size=self.number_of_processes,
                                                                                         learn_model=learn_model,
                                                                                         encode_labels=True),
                                                            ))

        return encoded_dataset

    def _get_label_config(self, dataset):
        label_config = LabelHelper.create_label_config([self.label], dataset, MotifGeneralizationAnalysis.__name__,
                                                       f"{MotifGeneralizationAnalysis.__name__}/label")
        EncoderHelper.check_positive_class_labels(label_config, f"{MotifGeneralizationAnalysis.__name__}/label")

        return label_config

    def _write_output_tables(self, training_plotting_data, test_plotting_data, training_combined_precision, test_combined_precision):
        results_table_name = "Confusion matrix and precision/recall scores for each significant motif on the {} set"
        combined_precision_table_name = "Combined precision scores on the {} set for each TP value on the training set"

        train_results_table = self._write_output_table(training_plotting_data, self.result_path / f"training_set_scores.csv", results_table_name.format("training"))
        test_results_table = self._write_output_table(test_plotting_data, self.result_path / f"test_set_scores.csv", results_table_name.format("test"))
        training_combined_precision_table = self._write_output_table(training_combined_precision, self.result_path / f"training_combined_precision.csv", combined_precision_table_name.format("training"))
        test_combined_precision_table = self._write_output_table(test_combined_precision, self.result_path / f"test_combined_precision.csv", combined_precision_table_name.format("test"))

        return [table for table in [train_results_table, test_results_table, training_combined_precision_table, test_combined_precision_table] if table is not None]

    def _write_stats(self):
        with open(self.result_path / f"info.txt", "w") as file:
            file.writelines([f"total training+test size: {self.dataset.get_example_count()}\n",
                             f"training TP cutoff: {self.tp_cutoff}\n",
                             f"training recall cutoff: {self.recall_cutoff}"])

    def _write_output_table(self, feature_annotations, file_path, name=None):
        feature_annotations.to_csv(file_path, index=False)

        return ReportOutput(
            path=file_path,
            name=name)

    def _write_plots(self, training_plotting_data, test_plotting_data, training_combined_precision, test_combined_precision):
        training_tp_plot = self._safe_plot(plotting_data=training_plotting_data, combined_precision=training_combined_precision, dataset_type="training set", file_path=self.result_path / f"training_precision_per_tp.html", plot_callable="_plot_precision_per_tp")
        test_tp_plot = self._safe_plot(plotting_data=test_plotting_data, combined_precision=test_combined_precision, dataset_type="test set", file_path=self.result_path / f"test_precision_per_tp.html", plot_callable="_plot_precision_per_tp")
        training_pr_plot = self._safe_plot(plotting_data=training_plotting_data, dataset_type="training set", plot_callable="_plot_precision_recall", file_path=self.result_path / f"training_precision_recall.html")
        test_pr_plot = self._safe_plot(plotting_data=test_plotting_data, dataset_type="test set", plot_callable="_plot_precision_recall", file_path=self.result_path / f"test_precision_recall.html")

        return [plot for plot in [training_tp_plot, test_tp_plot, training_pr_plot, test_pr_plot] if plot is not None]

    def _get_plotting_data(self, encoded_training_data, encoded_test_data):
        training_feature_annotations = self._annotate_precision_recall(encoded_training_data.encoded_data.feature_annotations)
        test_feature_annotations = self._annotate_precision_recall(encoded_test_data.encoded_data.feature_annotations)

        training_feature_annotations["training_tp_count"] = training_feature_annotations["raw_tp_count"]
        test_feature_annotations = self._get_merged_train_test_feature_annotations(training_feature_annotations,
                                                                                   test_feature_annotations)

        training_feature_annotations["highlight"] = self._get_highlight(training_feature_annotations)
        test_feature_annotations["highlight"] = self._get_highlight(test_feature_annotations)

        return training_feature_annotations, test_feature_annotations

    def _annotate_precision_recall(self, feature_annotations_table):
        feature_annotations_table = feature_annotations_table.copy()

        precision = self.col_names["precision"]
        recall = self.col_names["recall"]
        tp = self.col_names["tp"]
        fp = self.col_names["fp"]
        fn = self.col_names["fn"]

        feature_annotations_table[precision] = feature_annotations_table.apply(
            lambda row: 0 if row[tp] == 0 else row[tp] / (row[tp] + row[fp]), axis="columns")

        feature_annotations_table[recall] = feature_annotations_table.apply(
            lambda row: 0 if row[tp] == 0 else row[tp] / (row[tp] + row[fn]), axis="columns")

        return feature_annotations_table

    def _get_merged_train_test_feature_annotations(self, training_feature_annotations, test_feature_annotations):
        training_info_to_merge = training_feature_annotations[["feature_names", "training_tp_count"]].copy()
        test_info_to_merge = test_feature_annotations.copy()

        merged_train_test_info = training_info_to_merge.merge(test_info_to_merge)

        return merged_train_test_info

    def _get_highlight(self, feature_annotations):
        if self.highlight_motifs_path is not None:
            highlight_motifs = [PositionalMotifHelper.motif_to_string(indices, amino_acids, motif_sep="-", newline=False)
                                for indices, amino_acids in PositionalMotifHelper.read_motifs_from_file(self.highlight_motifs_path)]

            return [self.highlight_motifs_name if motif in highlight_motifs else "Motif" for motif in feature_annotations["feature_names"]]
        else:
            return ["Motif"] * len(feature_annotations)

    def _get_combined_precision(self, plotting_data):
        group_by_tp = plotting_data.groupby("training_tp_count")

        tp, fp = self.col_names["tp"], self.col_names["fp"]
        combined_precision = group_by_tp[tp].sum() / (group_by_tp[tp].sum() + group_by_tp[fp].sum())

        df = pd.DataFrame({"training_tp": list(combined_precision.index),
                           "combined_precision": list(combined_precision)})

        if self.smoothen_combined_precision:
            df["smooth_combined_precision"] = self._smooth_combined_precision(list(combined_precision.index),
                                                                              list(combined_precision),
                                                                              list(group_by_tp[tp].count()))

        return df

    def _set_tp_cutoff(self, combined_precision):
        col = "smooth_combined_precision" if "smooth_combined_precision" in combined_precision.columns else "combined_precision"

        above_threshold = combined_precision[combined_precision[col] > self.min_precision]
        if len(above_threshold) > 0:
            self.tp_cutoff = min(above_threshold["training_tp"])
        else:
            warnings.warn(f"{MotifGeneralizationAnalysis.__name__}: could not automatically determine optimal TP threshold, {col} did not reach minimal preciison {self.min_precision}")
            self.tp_cutoff = None

    def _get_positive_count(self, dataset):
        label_config = self._get_label_config(dataset)
        label_name = label_config.get_label_objects()[0].name
        label_positive_class = label_config.get_label_objects()[0].positive_class

        return sum([1 for label_class in dataset.get_metadata([label_name])[label_name] if label_class == label_positive_class])

    def _set_recall_cutoff(self, encoded_training_data):
        positive_count_training = self._get_positive_count(encoded_training_data)
        if self.tp_cutoff is not None:
            self.recall_cutoff = self.tp_cutoff / positive_count_training

    def _smooth_combined_precision(self, x, y, weights):
        smoothed_y = []

        for i in range(len(x)):
            scale = self._get_lognorm_scale(x, i, weights)

            lognorm_for_this_x = lognorm.pdf(x, s=0.1, loc=x[i] - scale, scale=scale)

            smoothed_y.append(sum(lognorm_for_this_x * y) / sum(lognorm_for_this_x))

        return smoothed_y

    def _get_lognorm_scale(self, x, i, weights):
        window_size = self._determine_window_size(x, i, weights)
        return window_size * self.smoothing_constant1 + self.smoothing_constant2

    def _determine_window_size(self, x, i, weights):
        x_rng = 0
        n_data_points = weights[i]

        assert sum(
            weights) > self.min_points_in_window, f"{self.__class__.__name__}: min_points_in_window ({self.min_points_in_window}) is smaller than the total number of points in the plot ({sum(weights)}). Please decrease the value for min_points_in_window. Skipping this plot..."

        while n_data_points < self.min_points_in_window:
            x_rng += 1

            to_select = [j for j in range(len(x)) if (x[i] - x_rng) <= x[j] <= (x[i] + x_rng)]
            lower_index = min(to_select)
            upper_index = max(to_select)

            n_data_points = sum(weights[lower_index:upper_index + 1])

        return x_rng

    def _plot_precision_per_tp(self, file_path, plotting_data, combined_precision, dataset_type):
        fig = px.strip(plotting_data,
                       y=self.col_names["precision"], x="training_tp_count", hover_data=["feature_names"],
                       range_y=[0, 1.01], color="highlight",
                       color_discrete_map={"Motif": "#74C4C4",
                                           self.highlight_motifs_name: px.colors.qualitative.Pastel[1]},
                       stripmode='overlay', log_x=True,
                       labels={
                           "precision_scores": f"Precision ({dataset_type})",
                           "weighted_precision_scores": f"Weighted precision ({dataset_type})",
                           "feature_names": "Motif",
                           "raw_tp_count": "True positive predictions (training set)"
                       })

        fig.add_trace(go.Scatter(x=combined_precision["training_tp"], y=combined_precision["combined_precision"],
                                 mode='markers+lines', name=self.col_names["combined precision"],
                                 marker=dict(symbol="diamond", color=px.colors.diverging.Tealrose[0])),
                      secondary_y=False)

        if self.smoothen_combined_precision:
            fig.add_trace(go.Scatter(x=combined_precision["training_tp"], y=combined_precision["smooth_combined_precision"],
                                     marker=dict(color=px.colors.diverging.Tealrose[-1]),
                                     name=self.col_names["combined precision"] + ", smoothed",
                                     mode="lines", line_shape='spline', line={'smoothing': 1.3}),
                          secondary_y=False, )

        if self.tp_cutoff is not None and dataset_type == "test set":
            fig.add_vline(x=self.tp_cutoff, line_dash="dash")

        fig.update_layout(xaxis=dict(dtick=1), showlegend=True)

        fig.write_html(str(file_path))

        return ReportOutput(
            path=file_path,
            name=f"Precision scores on the {dataset_type} for motifs found at each true positive count of the training set.",
        )

    def _plot_precision_recall(self, file_path, plotting_data, min_recall=None, min_precision=None, dataset_type=None):
        fig = px.scatter(plotting_data,
                         y=self.col_names["precision"], x=self.col_names["recall"], hover_data=["feature_names"],
                         range_x=[0, 1.01], range_y=[0, 1.01], color="highlight",
                         color_discrete_map={"Motif": px.colors.qualitative.Pastel[0],
                                             self.highlight_motifs_name: px.colors.qualitative.Pastel[1]},
                         labels={
                             "precision_scores": f"Precision ({dataset_type})",
                             "recall_scores": f"Recall ({dataset_type})",
                             "weighted_precision_scores": f"Weighted precision ({dataset_type})",
                             "weighted_recall_scores": f"Weighted recall ({dataset_type})",
                             "feature_names": "Motif",
                         })

        if min_precision is not None and min_precision > 0:
            fig.add_hline(y=min_precision, line_dash="dash")

        if min_recall is not None and min_recall > 0:
            fig.add_vline(x=min_recall, line_dash="dash")

        fig.write_html(str(file_path))

        return ReportOutput(
            path=file_path,
            name=f"Precision versus recall of significant motifs on the {dataset_type}",
        )