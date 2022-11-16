from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

from typing import List

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

        n_splits (int): number of random data splits

        label (dict): A label configuration. One label should be specified, and the positive_class for this label should be defined. See the YAML specification below for an example.


        highlight_motifs_path (str): path to a set of motifs of interest to highlight in the output figures. By default no motifs are highlighted.

        highlight_motifs_name (str): if highlight_motifs_path is defined, this name will be used to label the motifs of interest in the output figures.

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

    def __init__(self, training_percentage: float = None, max_positions: int = None, min_precision: float = None,
                 min_recall: float = None, min_true_positives: int = None,  random_seed: int = None, label: dict = None,
                 highlight_motifs_path: str = None, highlight_motifs_name: str = None,
                 dataset: SequenceDataset = None, result_path: Path = None, number_of_processes: int = 1, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.training_percentage = training_percentage
        self.max_positions = max_positions
        self.max_positions = max_positions
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.min_true_positives = min_true_positives
        self.random_seed = random_seed
        self.label = label
        self.label_config = None
        self.weighted = None

        self.highlight_motifs_name = highlight_motifs_name
        self.highlight_motifs_path = Path(highlight_motifs_path) if highlight_motifs_path is not None else None

    @classmethod
    def build_object(cls, **kwargs):
        location = MotifGeneralizationAnalysis.__name__

        ParameterValidator.assert_type_and_value(kwargs["training_percentage"], float, location, "training_percentage", min_exclusive=0, max_exclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["max_positions"], int, location, "max_positions", min_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["min_precision"], (int, float), location, "min_precision", min_inclusive=0, max_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["min_recall"], (int, float), location, "min_recall", min_inclusive=0, max_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["min_true_positives"], int, location, "min_true_positives", min_inclusive=1)

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
        self.weighted = True if self.dataset.get_example_weights() is not None else False

        encoded_training_data, encoded_test_data = self._get_encoded_train_test_data()

        training_feature_annotations = self._annotate_precision_recall(encoded_training_data.encoded_data.feature_annotations)
        test_feature_annotations = self._annotate_precision_recall(encoded_test_data.encoded_data.feature_annotations)

        train_results_table = self._write_output_table(training_feature_annotations, self.result_path / f"training_set_scores.csv", "training set")
        test_results_table = self._write_output_table(test_feature_annotations, self.result_path / f"test_set_scores.csv", "test set")

        results_plots = self._safe_plot(training_feature_annotations=training_feature_annotations,
                                        test_feature_annotations=test_feature_annotations)

        return ReportResult(output_tables=[table for table in [train_results_table, test_results_table] if table is not None],
                            output_figures=results_plots)

    def _annotate_precision_recall(self, feature_annotations_table):
        feature_annotations_table = feature_annotations_table.copy()

        if self.weighted:
            tp, fp, fn = "weighted_tp_count", "weighted_fp_count", "weighted_fn_count"
            precision, recall = "weighted_precision_scores", "weighted_recall_scores"
        else:
            tp, fp, fn = "raw_tp_count", "raw_fp_count", "raw_fn_count"
            precision, recall = "precision_scores", "recall_scores"

        feature_annotations_table[precision] = feature_annotations_table.apply(
            lambda row: 0 if row[tp] == 0 else row[tp] / (row[tp] + row[fp]), axis="columns")

        feature_annotations_table[recall] = feature_annotations_table.apply(
            lambda row: 0 if row[tp] == 0 else row[tp] / (row[tp] + row[fn]), axis="columns")

        return feature_annotations_table


    def _get_encoded_train_test_data(self):
        train_data_path = PathBuilder.build(self.result_path / "datasets/train")
        test_data_path = PathBuilder.build(self.result_path / "datasets/test")

        train_indices, val_indices = Util.get_train_val_indices(self.dataset.get_example_count(),
                                                                self.training_percentage, random_seed=self.random_seed)
        training_data = self.dataset.make_subset(train_indices, train_data_path, Dataset.TRAIN)
        test_data = self.dataset.make_subset(val_indices, test_data_path, Dataset.TEST)

        encoder = self._get_encoder()

        encoded_training_data = self._encode_dataset(training_data, encoder, learn_model=True)
        encoded_test_data = self._encode_dataset(test_data, encoder, learn_model=False)

        return encoded_training_data, encoded_test_data

    def _write_results_tables(self, training_feature_annotations, test_feature_annotations):

        train_results_table = self._write_output_table(training_feature_annotations, self.result_path  / f"training_set_scores.csv", "training set")
        test_results_table = self._write_output_table(test_feature_annotations, self.result_path  / f"test_set_scores.csv", "test set")

        return [table for table in [train_results_table, test_results_table] if table is not None]

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

    def _get_result_suffix(self, dataset_type):
        return f" on the {dataset_type}" if dataset_type is not None else ""

    def _write_output_table(self, feature_annotations, file_path, dataset_type=None):
        feature_annotations.to_csv(file_path, index=False)

        return ReportOutput(
            path=file_path,
            name=f"Confusion matrix and precision/recall scores for each significant motif{self._get_result_suffix(dataset_type)}",
        )

    def _get_color(self, feature_annotations):
        if self.highlight_motifs_path is not None:
            highlight_motifs = [PositionalMotifHelper.motif_to_string(indices, amino_acids, motif_sep="-", newline=False)
                                for indices, amino_acids in PositionalMotifHelper.read_motifs_from_file(self.highlight_motifs_path)]

            return [self.highlight_motifs_name if motif in highlight_motifs else "Significant motif" for motif in
                    feature_annotations["feature_names"]]

    def _plot_precision_recall(self, file_path, feature_annotations, min_recall=None, min_precision=None, dataset_type=None):
        y = "weighted_precision_scores" if self.weighted else "precision_scores"
        x = "weighted_recall_scores" if self.weighted else "recall_scores"

        fig = px.scatter(feature_annotations,
                         y=y, x=x, hover_data=["feature_names"],
                         range_x=[0, 1], range_y=[0, 1], color=self._get_color(feature_annotations),
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         labels={
                             "precision_scores": f"Precision ({dataset_type})",
                             "recall_scores": f"Recall ({dataset_type})",
                             "weighted_precision_scores": f"Weighted precision ({dataset_type})",
                             "weighted_recall_scores": f"Weighted recall ({dataset_type})",
                             "feature_names": "Motif"
                         })

        if min_precision is not None and min_precision > 0:
            fig.add_hline(y=min_precision, line_dash="dash")

        if min_recall is not None and min_recall > 0:
            fig.add_vline(x=min_recall, line_dash="dash")

        fig.write_html(str(file_path))

        return ReportOutput(
            path=file_path,
            name=f"Precision versus recall of significant motifs{self._get_result_suffix(dataset_type)}",
        )

    def _get_dataset_type_prefix(self, dataset_type):
        assert dataset_type in ["training set", "test set"]
        return "train_" if dataset_type == "training set" else "test_"

    def _get_pr_plot(self, feature_annotations, dataset_type):
        prefix = self._get_dataset_type_prefix(dataset_type)

        return self._plot_precision_recall(self.result_path / f"{prefix}precision_recall.html",
                                            feature_annotations,
                                            min_recall=self.min_recall,
                                            min_precision=0,
                                            dataset_type=dataset_type)

    def _plot_precision_per_tp(self, file_path, feature_annotations, dataset_type):
        y = "weighted_precision_scores" if self.weighted else "precision_scores"
        tp, fp = ("weighted_tp_count", "weighted_fp_count") if self.weighted else ("raw_tp_count", "raw_fp_count")

        fig = px.strip(feature_annotations,
                       y=y, x="training_tp_count", hover_data=["feature_names"],
                       range_y=[0, 1], color=self._get_color(feature_annotations),
                       color_discrete_sequence=px.colors.qualitative.Pastel,
                       stripmode='overlay', log_x=True,
                       labels={
                           "precision_scores": f"Precision ({dataset_type})",
                           "weighted_precision_scores": f"Weighted precision ({dataset_type})",
                           "feature_names": "Motif",
                           "raw_tp_count": "True positive predictions (training set)"
                       })

        # mean_precision = feature_annotations.groupby("raw_tp_count")[y].mean()
        group_by_tp = feature_annotations.groupby("training_tp_count")
        combined_precision = group_by_tp[tp].sum() / (group_by_tp[tp].sum() + group_by_tp[fp].sum())

        fig.add_trace(go.Scatter(x=list(combined_precision.index), y=list(combined_precision),
                                 marker=dict(color=px.colors.diverging.Tealrose[0])), secondary_y=False)

        fig.update_layout(showlegend=False)

        fig.update_layout(
            xaxis=dict(
                # tickmode='linear',
                dtick=1
            )
        )

        fig.write_html(str(file_path))

        return ReportOutput(
            path=file_path,
            name=f"Precision scores on the {self._get_result_suffix(dataset_type)} for motifs found at each true positive count of the training set.",
        )

    def _get_tp_plot(self, feature_annotations, dataset_type):
        prefix = self._get_dataset_type_prefix(dataset_type)

        return self._plot_precision_per_tp(file_path=self.result_path / f"{prefix}precision_per_tp.html",
                                           feature_annotations=feature_annotations,
                                           dataset_type=dataset_type)

    def _get_merged_train_test_feature_annotations(self, training_feature_annotations, test_feature_annotations):
        precision_colname = "weighted_precision_scores" if self.weighted else "precision_scores"
        tp_colname, fp_colname = ("weighted_tp_count", "weighted_fp_count") if self.weighted else ("raw_tp_count", "raw_fp_count")

        training_info_to_merge = training_feature_annotations[["feature_names", "training_tp_count"]].copy()
        test_info_to_merge = test_feature_annotations[["feature_names", tp_colname, fp_colname, precision_colname]].copy()

        merged_train_test_info = training_info_to_merge.merge(test_info_to_merge)

        # merged_train_test_info = merged_train_test_info.loc[merged_train_test_info[tp_colname] != 0]

        return merged_train_test_info


    def _plot(self, training_feature_annotations, test_feature_annotations) -> List[ReportOutput]:
        training_pr_plot = self._get_pr_plot(training_feature_annotations, "training set")
        test_pr_plot = self._get_pr_plot(test_feature_annotations, "test set")

        training_feature_annotations["training_tp_count"] = training_feature_annotations["raw_tp_count"]
        training_tp_plot = self._get_tp_plot(training_feature_annotations, "training set")

        merged_train_test_feature_annotations = self._get_merged_train_test_feature_annotations(training_feature_annotations, test_feature_annotations)

        test_tp_plot = self._get_tp_plot(merged_train_test_feature_annotations, "test set")

        return [plot for plot in [training_pr_plot, test_pr_plot, training_tp_plot, test_tp_plot] if plot is not None]
