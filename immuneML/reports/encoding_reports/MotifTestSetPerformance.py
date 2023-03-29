import logging
import warnings
from pathlib import Path
import shutil

import numpy as np

from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.dsl.import_parsers.ImportParser import ImportParser
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.encodings.motif_encoding.PositionalMotifHelper import PositionalMotifHelper
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.ImportHelper import ImportHelper
from immuneML.util.MotifPerformancePlotHelper import MotifPerformancePlotHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


class MotifTestSetPerformance(EncodingReport):
    """
    This report can be used to show the performance of a learned set motifs using the :py:obj:`~immuneML.encodings.motif_encoding.MotifEncoder.MotifEncoder`
    on an independent test set of unseen data.

    It is recommended to first run the report :py:obj:`~immuneML.reports.data_reports.MotifGeneralizationAnalysis.MotifGeneralizationAnalysis`
    in order to calibrate the optimal recall thresholds and plot the performance of motifs on training- and validation sets.

    Arguments:

        test_dataset (dict): parameters for importing a SequenceDataset to use as an independent test set. By default,
        the import parameters 'is_repertoire' and 'paired' will be set to False to ensure a SequenceDataset is imported.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_motif_report:
            MotifTestSetPerformance:
                test_dataset:
                    format: AIRR # choose any valid import format
                    params:
                        path: path/to/files/
                        is_repertoire: False  # is_repertoire must be False to import a SequenceDataset
                        paired: False         # paired must be False to import a SequenceDataset
                        # optional other parameters...

    """

    def __init__(self, dataset: Dataset = None, result_path: Path = None, test_dataset_import_cls: DataImport = None,
                 test_dataset_import_params: DatasetImportParams = None,
                 training_set_name: str = None, test_set_name: str = None,
                 split_by_motif_size: bool = None,
                 highlight_motifs_path: str = None, highlight_motifs_name: str = None,
                 min_points_in_window: int = None,
                 smoothing_constant1: float = None, smoothing_constant2: float = None,
                 keep_test_dataset: bool = None,
                 number_of_processes: int = 1, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.test_dataset_import_cls = test_dataset_import_cls
        self.test_dataset_import_params = test_dataset_import_params

        self.keep_test_dataset = keep_test_dataset
        self.split_by_motif_size = split_by_motif_size
        self.training_set_name = training_set_name
        self.test_set_name = test_set_name
        self.highlight_motifs_path = highlight_motifs_path
        self.highlight_motifs_name = highlight_motifs_name
        self.min_points_in_window = min_points_in_window
        self.smoothing_constant1 = smoothing_constant1
        self.smoothing_constant2 = smoothing_constant2

    @classmethod
    def build_object(cls, **kwargs):
        location = MotifTestSetPerformance.__name__

        import_cls, test_dataset_import_params = MotifTestSetPerformance._parse_dataset_params(kwargs)

        kwargs["test_dataset_import_cls"] = import_cls
        kwargs["test_dataset_import_params"] = test_dataset_import_params

        del kwargs["test_dataset"]

        ParameterValidator.assert_type_and_value(kwargs["split_by_motif_size"], bool, location, "split_by_motif_size")
        ParameterValidator.assert_type_and_value(kwargs["training_set_name"], str, location, "training_set_name")
        ParameterValidator.assert_type_and_value(kwargs["test_set_name"], str, location, "test_set_name")
        ParameterValidator.assert_type_and_value(kwargs["keep_test_dataset"], bool, location, "keep_test_dataset")
        ParameterValidator.assert_type_and_value(kwargs["min_points_in_window"], int, location, "min_points_in_window", min_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["smoothing_constant1"], (int, float), location, "smoothing_constant1", min_exclusive=0)
        ParameterValidator.assert_type_and_value(kwargs["smoothing_constant2"], (int, float), location, "smoothing_constant2", min_exclusive=0)

        if "highlight_motifs_path" in kwargs and kwargs["highlight_motifs_path"] is not None:
            PositionalMotifHelper.check_motif_filepath(kwargs["highlight_motifs_path"], location, "highlight_motifs_path")

        ParameterValidator.assert_type_and_value(kwargs["highlight_motifs_name"], str, location, "highlight_motifs_name")

        return MotifTestSetPerformance(**kwargs)

    @staticmethod
    def _parse_dataset_params(kwargs):
        location = MotifTestSetPerformance.__name__

        ParameterValidator.assert_type_and_value(kwargs["test_dataset"], dict, location, "test_dataset")
        ParameterValidator.assert_keys_present(kwargs["test_dataset"].keys(), ["format", "params"], location,
                                               "test_dataset")
        ParameterValidator.assert_type_and_value(kwargs["test_dataset"]["format"], str, location, "test_dataset/format")

        import_cls = ReflectionHandler.get_class_by_name("{}Import".format(kwargs["test_dataset"]["format"]))
        default_params = DefaultParamsLoader.load(ImportParser.keyword, kwargs["test_dataset"]["format"])
        params_dict = {**default_params, **kwargs["test_dataset"]["params"]}

        test_dataset_import_params = DatasetImportParams.build_object(**params_dict)

        if test_dataset_import_params.is_repertoire:
            warnings.warn(f"{location}: This report only allows the reference dataset to be of type SequenceDataset. "
                          "Setting 'test_dataset/params/is_repertoire' to False...")
            test_dataset_import_params.is_repertoire = False

        if test_dataset_import_params.paired:
            warnings.warn(f"{location}: This report only allows the reference dataset to be of type SequenceDataset. "
                          "Setting 'test_dataset/params/paired' to False...")
            test_dataset_import_params.paired = False

        assert test_dataset_import_params.metadata_column_mapping is not None, f"{location}: This report requires a test_dataset containing the same label as the encoded dataset. Please set a label using 'test_dataset/params/metadata_column_mapping'."

        return import_cls, test_dataset_import_params

    def _generate(self) -> ReportResult:
        test_dataset = self._get_test_dataset()
        test_encoded_data = self._encode_test_data(test_dataset)

        training_plotting_data, test_plotting_data = MotifPerformancePlotHelper.get_plotting_data(self.dataset.encoded_data,
                                                                                                  test_encoded_data.encoded_data,
                                                                                                  self.highlight_motifs_path,
                                                                                                  self.highlight_motifs_name)

        training_plotting_data["motif_size"] = training_plotting_data["feature_names"].apply(PositionalMotifHelper.get_motif_size)
        test_plotting_data["motif_size"] = test_plotting_data["feature_names"].apply(PositionalMotifHelper.get_motif_size)

        output_tables, output_plots = self._get_report_outputs(training_plotting_data, test_plotting_data)

        if not self.keep_test_dataset:
            shutil.rmtree(self.test_dataset_import_params.result_path)

        return ReportResult(name=self.name,
                            info="Performance of motifs on an independent test set",
                            output_figures=output_plots,
                            output_tables=output_tables)

    def _get_report_outputs(self, training_plotting_data, test_plotting_data):
        if self.split_by_motif_size:
            return self._construct_and_plot_data_per_motif_size(training_plotting_data, test_plotting_data)
        else:
            return self._construct_and_plot_data(training_plotting_data, test_plotting_data)


    def _construct_and_plot_data_per_motif_size(self, training_plotting_data, test_plotting_data):
        output_tables, output_plots = [], []

        for motif_size in sorted(set(training_plotting_data["motif_size"])):
            sub_training_plotting_data = training_plotting_data[training_plotting_data["motif_size"] == motif_size]
            sub_test_plotting_data = test_plotting_data[test_plotting_data["motif_size"] == motif_size]

            sub_output_tables, sub_output_plots = self._construct_and_plot_data(sub_training_plotting_data, sub_test_plotting_data, motif_size=motif_size)

            output_tables.extend(sub_output_tables)
            output_plots.extend(sub_output_plots)

        return output_tables, output_plots
    def _construct_and_plot_data(self, training_plotting_data, test_plotting_data, motif_size=None):
        training_combined_precision = self._get_combined_precision(training_plotting_data)
        test_combined_precision = self._get_combined_precision(test_plotting_data)

        motif_size_suffix = f"_motif_size={motif_size}" if motif_size is not None else ""
        motifs_name = f"motifs of length {motif_size}" if motif_size is not None else "motifs"

        output_tables = MotifPerformancePlotHelper.write_output_tables(self, training_plotting_data, test_plotting_data, training_combined_precision, test_combined_precision, motifs_name=motifs_name, file_suffix=motif_size_suffix)
        output_plots = MotifPerformancePlotHelper.write_plots(self, training_plotting_data, test_plotting_data, training_combined_precision, test_combined_precision, training_tp_cutoff="auto", test_tp_cutoff="auto", motifs_name=motifs_name, file_suffix=motif_size_suffix)

        return output_tables, output_plots

    def _get_combined_precision(self, plotting_data):
        return MotifPerformancePlotHelper.get_combined_precision(plotting_data,
                                                                 min_points_in_window=self.min_points_in_window,
                                                                 smoothing_constant1=self.smoothing_constant1,
                                                                 smoothing_constant2=self.smoothing_constant2)

    def _write_output_tables(self, training_plotting_data, test_plotting_data, training_combined_precision, test_combined_precision, file_suffix=""):
        results_table_name = "Confusion matrix and precision/recall scores for significant motifs on the {}"
        combined_precision_table_name = "Combined precision scores of motifs on the {} for each TP value on the " + str(self.training_set_name)

        train_results_table = self._write_output_table(training_plotting_data, self.result_path / f"training_set_scores{file_suffix}.csv", results_table_name.format(self.training_set_name))
        test_results_table = self._write_output_table(test_plotting_data, self.result_path / f"test_set_scores{file_suffix}.csv", results_table_name.format(self.test_set_name))
        training_combined_precision_table = self._write_output_table(training_combined_precision, self.result_path / f"training_combined_precision{file_suffix}.csv", combined_precision_table_name.format(self.training_set_name))
        test_combined_precision_table = self._write_output_table(test_combined_precision, self.result_path / f"test_combined_precision{file_suffix}.csv", combined_precision_table_name.format(self.test_set_name))

        return [table for table in [train_results_table, test_results_table, training_combined_precision_table,
                                    test_combined_precision_table] if table is not None]

    def _plot_precision_per_tp(self, file_path, plotting_data, combined_precision, dataset_type, tp_cutoff, motifs_name="motifs"):
        return MotifPerformancePlotHelper.plot_precision_per_tp(file_path, plotting_data, combined_precision, dataset_type,
                                                                training_set_name=self.training_set_name,
                                                                motifs_name=motifs_name,
                                                                tp_cutoff=tp_cutoff,
                                                                highlight_motifs_name=self.highlight_motifs_name)

    def _plot_precision_recall(self, file_path, plotting_data, min_recall=None, min_precision=None, dataset_type=None, motifs_name="motifs"):
        return MotifPerformancePlotHelper.plot_precision_recall(file_path, plotting_data, min_recall=min_recall, min_precision=min_precision,
                                                                dataset_type=dataset_type, motifs_name=motifs_name, highlight_motifs_name=self.highlight_motifs_name)

    def _encode_test_data(self, test_dataset):
        encoder = self._get_encoder()
        params = EncoderParams(result_path=self.result_path / "encoded_test_dataset",
                               label_config=self._get_label_config(),
                               pool_size=self.number_of_processes,
                               learn_model=False)

        return encoder.encode(test_dataset, params)

    def _get_encoder(self):
        encoder = MotifEncoder(label=self._get_label_name(),
                               name=f"motif_encoder_{self.name}")

        encoder.learned_motif_filepath = self.dataset.encoded_data.info["learned_motif_filepath"]

        return encoder


    def _get_test_dataset_y_true(self, test_dataset):
        label_name = self._get_label_name()
        positive_class = self._get_positive_class()

        y_true = [sequence.get_attribute(label_name) == positive_class for sequence in test_dataset.get_data()]

        return np.array(y_true)

    def _get_motifs(self):
        motif_names = self._get_motif_names()
        return [PositionalMotifHelper.string_to_motif(name, "&", "-") for name in motif_names]

    def _get_motif_names(self):
        return list(self.dataset.encoded_data.feature_annotations.feature_names)

    def _get_test_dataset(self):
        self._set_result_path()
        test_dataset = self._import_test_dataset()
        self._check_test_dataset(test_dataset)

        return test_dataset

    def _check_test_dataset(self, test_dataset):
        self._check_sequence_length(test_dataset)
        self._check_dataset_label(test_dataset)

    def _check_sequence_length(self, test_dataset):
        legal_length = self._get_legal_sequence_length()

        for sequence in test_dataset.get_data():
            assert len(sequence.get_sequence()) == legal_length, f"{MotifTestSetPerformance.__name__}: the length of the sequences in the test dataset is required to match the length of the original dataset ({legal_length}). Found sequence of length: {len(sequence.get_sequence())}"

    def _get_legal_sequence_length(self):
        sequence = next(self.dataset.get_data())

        return len(sequence.get_sequence())

    def _check_dataset_label(self, test_dataset):
        label_name = self._get_label_name()
        label_values = set(self.dataset.encoded_data.labels[label_name])

        error = f"{self.__class__.__name__}: expected only one label to be set for the test_dataset (label: {label_name}, with values: {', '.join(label_values)}). Instead found: {', '.join(test_dataset.get_label_names())}"

        assert len(test_dataset.get_label_names()) > 0, error + "no label set for the test_dataset."
        assert len(test_dataset.get_label_names()) == 1, error
        assert test_dataset.get_label_names() == [label_name], error

        test_dataset_label_values = set(test_dataset.get_metadata([label_name])[label_name])

        assert label_values == test_dataset_label_values, error + f" with classes {', '.join(test_dataset_label_values)}"

    def _get_label_config(self):
        return LabelConfiguration([self._get_label()])

    def _get_label(self):
        label_name = self._get_label_name()
        label_values = list(set(self.dataset.encoded_data.labels[label_name]))
        positive_class = self._get_positive_class()

        return Label(name=label_name, values=label_values, positive_class=positive_class)

    def _get_label_name(self):
        return list(self.dataset.encoded_data.labels.keys())[0]

    def _get_positive_class(self):
        return self.dataset.encoded_data.info["positive_class"]

    def _set_result_path(self):
        self.test_dataset_import_params.result_path = self.result_path / f"test_dataset_{self.name}"

    def _import_test_dataset(self):
        return ImportHelper.import_sequence_dataset(self.test_dataset_import_cls, self.test_dataset_import_params, f"test_dataset_{self.name}")

    def check_prerequisites(self) -> bool:
        location = MotifTestSetPerformance.__name__

        if self.dataset.encoded_data is None or self.dataset.encoded_data.info is None:
            logging.warning(f"{location}: the dataset is not encoded, skipping this report...")
            return False
        elif self.dataset.encoded_data.encoding != MotifEncoder.__name__:
            logging.warning(
                f"{location}: the dataset encoding ({self.dataset.encoded_data.encoding}) "
                f"does not match the required encoding ({MotifEncoder.__name__}), skipping this report...")
            return False
        elif self.dataset.encoded_data.feature_annotations is None:
            logging.warning(f"{location}: missing feature annotations for {MotifEncoder.__name__},"
                            f"skipping this report...")
            return False
        else:
            return True
