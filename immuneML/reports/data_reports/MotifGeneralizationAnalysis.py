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
from immuneML.util.MotifPerformancePlotHelper import MotifPerformancePlotHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.steps.DataEncoder import DataEncoder
from immuneML.workflows.steps.DataEncoderParams import DataEncoderParams


class MotifGeneralizationAnalysis(DataReport):
    """
    This report splits the given dataset into a training and validation set, identifies significant motifs using the
    :py:obj:`~immuneML.encodings.motif_encoding.MotifEncoder.MotifEncoder`
    on the training set and plots the precision/recall and precision/true positive predictions of motifs
    on both the training and validation sets. This can be used to:
    - determine the optimal recall cutoff for motifs of a given size
    - investigate how well motifs learned on a training set generalize to a test set

    After running this report and determining the optimal recall cutoffs, the report
    :py:obj:`~immuneML.reports.encoding_reports.MotifTestSetPerformance.MotifTestSetPerformance` can be run to
    plot the performance on an independent test set.

    Arguments:

        label (dict): A label configuration. One label should be specified, and the positive_class for this label should be defined. See the YAML specification below for an example.

        highlight_motifs_path (str): path to a set of motifs of interest to highlight in the output figures. By default no motifs are highlighted.

        highlight_motifs_name (str): if highlight_motifs_path is defined, this name will be used to label the motifs of interest in the output figures.

        smoothen_combined_precision (bool): whether to add a smoothed line representing the combined precision to the precision-vs-TP plot. When set to True, this may take considerable extra time to compute. By default, plot_smoothed_combined_precision is set to True.

        training_set_identifier_path (str): path to a file containing 'sequence_identifiers' of the sequences used for the training set. This file should have a single column named 'example_id' and have one sequence identifier per line. If training_set_identifier_path is not set, a random subset of the data (according to training_percentage) will be assigned to be the training set.


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
                 max_positions: int = None, min_precision: float = None, min_recall: float = None, min_true_positives: int = None,
                 test_precision_threshold: float = None,
                 split_by_motif_size: bool = None, random_seed: int = None, label: dict = None,
                 min_points_in_window: int = None, smoothing_constant1: float = None, smoothing_constant2: float = None,
                 highlight_motifs_path: str = None, highlight_motifs_name: str = None,
                 training_set_name: str = None, test_set_name: str = None,
                 dataset: SequenceDataset = None, result_path: Path = None, number_of_processes: int = 1, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.training_set_identifier_path = Path(training_set_identifier_path) if training_set_identifier_path is not None else None
        self.training_percentage = training_percentage
        self.max_positions = max_positions
        self.max_positions = max_positions
        self.min_precision = min_precision
        self.test_precision_threshold = test_precision_threshold
        self.min_recall = min_recall
        self.min_true_positives = min_true_positives
        self.split_by_motif_size = split_by_motif_size
        self.min_points_in_window = min_points_in_window
        self.smoothing_constant1 = smoothing_constant1
        self.smoothing_constant2 = smoothing_constant2
        self.random_seed = random_seed
        self.label = label
        self.n_positives_in_training_data = None

        self.training_set_name = training_set_name
        self.test_set_name = test_set_name
        self.highlight_motifs_name = highlight_motifs_name
        self.highlight_motifs_path = Path(highlight_motifs_path) if highlight_motifs_path is not None else None

    @classmethod
    def build_object(cls, **kwargs):
        location = MotifGeneralizationAnalysis.__name__

        ParameterValidator.assert_type_and_value(kwargs["max_positions"], int, location, "max_positions", min_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["min_precision"], (int, float), location, "min_precision", min_inclusive=0, max_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["min_recall"], (int, float), location, "min_recall", min_inclusive=0, max_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["min_true_positives"], int, location, "min_true_positives", min_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["test_precision_threshold"], float, location, "test_precision_threshold", min_inclusive=0, max_exclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["split_by_motif_size"], bool, location, "split_by_motif_size")
        ParameterValidator.assert_type_and_value(kwargs["min_points_in_window"], int, location, "min_points_in_window", min_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["smoothing_constant1"], (int, float), location, "smoothing_constant1", min_exclusive=0)
        ParameterValidator.assert_type_and_value(kwargs["smoothing_constant2"], (int, float), location, "smoothing_constant2", min_exclusive=0)

        ParameterValidator.assert_type_and_value(kwargs["training_set_name"], str, location, "training_set_name")
        ParameterValidator.assert_type_and_value(kwargs["test_set_name"], str, location, "test_set_name")

        assert kwargs["training_set_name"] != kwargs["test_set_name"], f"{location}: training_set_name cannot be the same as test_set_name. Both are: {kwargs['training_set_name']}"

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
        encoded_training_dataset, encoded_test_dataset = self._get_encoded_train_test_datasets()
        training_plotting_data, test_plotting_data = MotifPerformancePlotHelper.get_plotting_data(encoded_training_dataset.encoded_data,
                                                                                                  encoded_test_dataset.encoded_data,
                                                                                                  self.highlight_motifs_path, self.highlight_motifs_name)

        self.n_positives_in_training_data = self._get_positive_count(encoded_training_dataset)

        return self._get_report_result(training_plotting_data, test_plotting_data)

    def _get_report_result(self, training_plotting_data, test_plotting_data):
        if self.split_by_motif_size:
            output_tables, output_texts, output_plots = self._construct_and_plot_data_per_motif_size(training_plotting_data, test_plotting_data)
        else:
            output_tables, output_texts, output_plots = self._construct_and_plot_data(training_plotting_data, test_plotting_data)

        return ReportResult(output_tables=output_tables,
                            output_text=output_texts,
                            output_figures=output_plots)

    def _construct_and_plot_data_per_motif_size(self, training_plotting_data, test_plotting_data):
        output_tables, output_texts, output_plots = [], [], []

        training_plotting_data["motif_size"] = training_plotting_data["feature_names"].apply(PositionalMotifHelper.get_motif_size)
        test_plotting_data["motif_size"] = test_plotting_data["feature_names"].apply(PositionalMotifHelper.get_motif_size)

        for motif_size in sorted(set(training_plotting_data["motif_size"])):
            sub_training_plotting_data = training_plotting_data[training_plotting_data["motif_size"] == motif_size]
            sub_test_plotting_data = test_plotting_data[test_plotting_data["motif_size"] == motif_size]

            sub_output_tables, sub_output_texts, sub_output_plots = self._construct_and_plot_data(sub_training_plotting_data, sub_test_plotting_data, motif_size=motif_size)

            output_tables.extend(sub_output_tables)
            output_texts.extend(sub_output_texts)
            output_plots.extend(sub_output_plots)

        return output_tables, output_texts, output_plots


    def _construct_and_plot_data(self, training_plotting_data, test_plotting_data, motif_size=None):
        training_combined_precision = self._get_combined_precision(training_plotting_data)
        test_combined_precision = self._get_combined_precision(test_plotting_data)
        tp_cutoff = self._determine_tp_cutoff(test_combined_precision, motif_size)
        recall_cutoff = self._tp_to_recall(tp_cutoff)

        motif_size_suffix = f"_motif_size={motif_size}" if motif_size is not None else ""
        motifs_name = f"motifs of lenght {motif_size}" if motif_size is not None else "motifs"

        output_tables = MotifPerformancePlotHelper.write_output_tables(self, training_plotting_data, test_plotting_data, training_combined_precision, test_combined_precision, motifs_name=motifs_name, file_suffix=motif_size_suffix)
        output_texts = self._write_stats(tp_cutoff, recall_cutoff, motifs_name=motifs_name, file_suffix=motif_size_suffix)
        output_plots = MotifPerformancePlotHelper.write_plots(self, training_plotting_data, test_plotting_data, training_combined_precision, test_combined_precision, tp_cutoff, motifs_name=motifs_name, file_suffix=motif_size_suffix)

        return output_tables, output_texts, output_plots

    def _get_encoded_train_test_datasets(self):
        train_data_path = PathBuilder.build(self.result_path / "datasets/train")
        test_data_path = PathBuilder.build(self.result_path / "datasets/test")

        train_indices, val_indices = self._get_train_val_indices()

        training_data = self.dataset.make_subset(train_indices, train_data_path, Dataset.TRAIN)
        test_data = self.dataset.make_subset(val_indices, test_data_path, Dataset.TEST)

        encoder = self._get_encoder()

        encoded_training_dataset = self._encode_dataset(training_data, encoder, learn_model=True)
        encoded_test_dataset = self._encode_dataset(test_data, encoder, learn_model=False)

        return encoded_training_dataset, encoded_test_dataset

    def _get_train_val_indices(self):
        if self.training_set_identifier_path is None:
            return Util.get_train_val_indices(self.dataset.get_example_count(),
                                              self.training_percentage, random_seed=self.random_seed)
        else:
            return self._get_train_val_indices_from_file()


    def _get_train_val_indices_from_file(self):
        input_train_identifiers = list(pd.read_csv(self.training_set_identifier_path, usecols=["example_id"])["example_id"].astype(str))

        train_indices = []
        val_indices = []
        val_identifiers = []
        actual_train_identifiers = []

        for idx, sequence in enumerate(self.dataset.get_data()):
            if str(sequence.identifier) in input_train_identifiers:
                train_indices.append(idx)
                actual_train_identifiers.append(sequence.identifier)
            else:
                val_indices.append(idx)
                val_identifiers.append(sequence.identifier)

        self._write_identifiers(self.result_path / "training_set_identifiers.txt", actual_train_identifiers, "Training")
        self._write_identifiers(self.result_path / "validation_set_identifiers.txt", val_identifiers, "Validation")

        assert len(train_indices) > 0, f"{MotifGeneralizationAnalysis.__name__}: error when reading training set identifiers from training_set_identifier_path, 0 of the identifiers were present in the dataset. Please check training_set_identifier_path: {self.training_set_identifier_path}, and see the log file for more information."
        assert len(val_indices) > 0, f"{MotifGeneralizationAnalysis.__name__}: error when inferring validation set identifiers from training_set_identifier_path, all of the identifiers were present in the dataset resulting in 0 sequences in the validation set. Please check training_set_identifier_path: {self.training_set_identifier_path}, and see the log file for more information."
        assert len(train_indices) == len(input_train_identifiers), f"{MotifGeneralizationAnalysis.__name__}: error when reading training set identifiers from training_set_identifier_path, not all identifiers provided in the file occurred in the dataset ({len(train_indices)} of {len(input_train_identifiers)} found). Please check training_set_identifier_path: {self.training_set_identifier_path}, and see the log file for more information."

        return train_indices, val_indices

    def _write_identifiers(self, path, identifiers, set_name):
        logging.info(f"{MotifGeneralizationAnalysis.__name__}: {len(identifiers)} {set_name} set identifiers written to: {path}")

        with open(path, "w") as file:
            file.writelines([f"{identifier}\n" for identifier in identifiers])

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

    def _get_positive_count(self, dataset):
        label_config = self._get_label_config(dataset)
        label_name = label_config.get_label_objects()[0].name
        label_positive_class = label_config.get_label_objects()[0].positive_class

        return sum([1 for label_class in dataset.get_metadata([label_name])[label_name] if label_class == label_positive_class])

    def _write_stats(self, tp_cutoff, recall_cutoff, motifs_name="all motifs", file_suffix=""):
        output_path = self.result_path / f"tp_recall_cutoffs{file_suffix}.txt"

        with open(output_path, "w") as file:
            file.writelines([f"total training+test size: {self.dataset.get_example_count()}\n",
                             f"total positives in training data: {self.n_positives_in_training_data}\n"
                             f"training TP cutoff: {tp_cutoff}\n",
                             f"training recall cutoff: {recall_cutoff}"])

        return [ReportOutput(path=output_path, name=f"TP and recall cutoffs for {motifs_name}")]

    def _get_combined_precision(self, plotting_data):
        return MotifPerformancePlotHelper.get_combined_precision(plotting_data,
                                                                 min_points_in_window=self.min_points_in_window,
                                                                 smoothing_constant1=self.smoothing_constant1,
                                                                 smoothing_constant2=self.smoothing_constant2)



    def _determine_tp_cutoff(self, combined_precision, motif_size=None):
        col = "smooth_combined_precision" if "smooth_combined_precision" in combined_precision else "combined_precision"

        try:
            # assert all(training_combined_precision["training_TP"] == test_combined_precision["training_TP"])
            #
            # train_test_difference = training_combined_precision[col] - test_combined_precision[col]
            # return min(test_combined_precision[train_test_difference <= self.precision_difference]["training_TP"])

            max_tp_below_threshold = max(combined_precision[combined_precision[col] < self.test_precision_threshold]["training_TP"])
            all_above_threshold = combined_precision[combined_precision["training_TP"] > max_tp_below_threshold]

            return min(all_above_threshold["training_TP"])
        except ValueError:
            motif_size_warning = f" for motif size = {motif_size}" if motif_size is not None else ""
            warnings.warn(f"{MotifGeneralizationAnalysis.__name__}: could not automatically determine optimal TP threshold{motif_size_warning} with precison differenc  based on {col}")
            return None

    def _tp_to_recall(self, tp_cutoff):
        if tp_cutoff is not None:
            return tp_cutoff / self.n_positives_in_training_data

    def _plot_precision_per_tp(self, file_path, plotting_data, combined_precision, dataset_type, tp_cutoff=None, motifs_name="motifs"):
         return MotifPerformancePlotHelper.plot_precision_per_tp(file_path, plotting_data, combined_precision, dataset_type,
                                                                 training_set_name=self.training_set_name,
                                                                 tp_cutoff=tp_cutoff, motifs_name=motifs_name,
                                                                 highlight_motifs_name=self.highlight_motifs_name)

    def _plot_precision_recall(self, file_path, plotting_data, min_recall=None, min_precision=None, dataset_type=None, motifs_name="motifs"):
        return MotifPerformancePlotHelper.plot_precision_recall(file_path, plotting_data, min_recall=min_recall, min_precision=min_precision,
                                                                 dataset_type=dataset_type, motifs_name=motifs_name, highlight_motifs_name=self.highlight_motifs_name)
