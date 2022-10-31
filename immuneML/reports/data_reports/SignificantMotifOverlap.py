import random
from pathlib import Path
import pandas as pd
import warnings
import logging
import numpy as np

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.dsl.instruction_parsers.LabelHelper import LabelHelper
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.motif_encoding.SignificantMotifEncoder import SignificantMotifEncoder
from immuneML.encodings.motif_encoding.PositionalMotifHelper import PositionalMotifHelper
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder

class SignificantMotifOverlap(DataReport):
    """
    This report splits the given dataset into n subsets, identifies significant motifs using the
    SignificantMotifEncoder for each of the data subsets, and computes the intersections between the significant
    motifs found in each pair of subsets. This can be used to investigate generalizability of motifs and calibrate
    example weighting parameters.

    Arguments:

        n_splits (int): number of random data splits

        label (dict): A label configuration. One label should be specified, and the positive_class for this label should be defined. See the YAML specification below for an example.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml


        my_report:
            SignificantMotifOverlap:
                ...
                label: # Define a label, and the positive class for that given label
                    CMV:
                        positive_class: +


    """

    def __init__(self, n_splits: int = None, max_positions: int = None, min_precision: float = None, min_recall: float = None,
                 min_true_positives: int = None, random_seed: int = None, label: dict = None,
                 dataset: Dataset = None, result_path: Path = None, number_of_processes: int = 1, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.n_splits = n_splits
        self.max_positions = max_positions
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.min_true_positives = min_true_positives
        self.random_seed = random_seed
        self.label = label
        self.label_config = None

    @classmethod
    def build_object(cls, **kwargs):
        location = SignificantMotifOverlap.__name__

        ParameterValidator.assert_type_and_value(kwargs["n_splits"], int, location, "n_splits", min_inclusive=2)
        ParameterValidator.assert_type_and_value(kwargs["max_positions"], int, location, "max_positions", min_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["min_precision"], (int, float), location, "min_precision", min_inclusive=0, max_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["min_recall"], (int, float), location, "min_recall", min_inclusive=0, max_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["min_true_positives"], int, location, "min_true_positives", min_inclusive=1)

        if "random_seed" in kwargs and kwargs["random_seed"] is not None:
            ParameterValidator.assert_type_and_value(kwargs["random_seed"], int, location, "random_seed")

        ParameterValidator.assert_type_and_value(kwargs["label"], dict, location, "label")
        assert len(kwargs["label"]) == 1, f"{location}: The number of specified labels must be 1, found {len(kwargs['label'])}: {', '.join(list(len(kwargs['label'].keys())))}"

        return SignificantMotifOverlap(**kwargs)

    def _generate(self) -> ReportResult:
        self.label_config = self._get_label_config()

        encoded_datasets = self._encode_datasets(self._get_splitted_dataset())
        # features_per_subset = self._get_significant_features_per_subset(encoded_datasets)

        result_files = self._write_result_files(encoded_datasets)

        return ReportResult(name=self.name,
                            info=f"Analysis of significant motifs found across {self.n_splits} subsets of the dataset.",
                            output_tables=result_files)


        # todo many-set overlap?
            # export all raw lists of amino acids
            # number of overlapping between each of them...
            # plot pairwise union sizes, see when it starts to go down?? -> multiple cutoffs... but we want to use one cutoff and multiple weighting params not multiple cutoffs
            # to see that with shitty parameterization there is no overlap, and better regularisation is more overlap
            # but we also find more motifs?? generally:number overlapping is good enough. also finding more motifs in all cases and having more overlapping is good.. right?
            # no also need to show total number found -> just in case you just find 100x more, but not more overlapping.. that would be bad. just as double check

    def _write_result_files(self, encoded_datasets):
        results_folder = PathBuilder.build(self.result_path / "feature_intersections")

        # features_per_subset = for encoded_dataset in encoded_datasets:
    #         if encoded_dataset is None:
    #             features_per_subset.append(set())
    #         else:
    #             features_per_subset.append(set(encoded_dataset.encoded_data.feature_names))


        subset_sizes_output = self._write_subset_sizes(encoded_datasets, results_folder)
        pairwise_intersection_output = self._write_pairwise_intersections(encoded_datasets, results_folder)
        multi_intersection_motifs = self._write_multi_intersection_motifs(encoded_datasets, results_folder)

        return [subset_sizes_output, pairwise_intersection_output, multi_intersection_motifs]

    def _get_motif_set(self, encoded_dataset):
        if encoded_dataset is None:
            return set()
        else:
            return set(encoded_dataset.encoded_data.feature_names)

    def _write_subset_sizes(self, encoded_datasets, results_folder):
        results_df = pd.DataFrame({"subset_idx": list(range(self.n_splits)),
                                   "number_of_motifs": [len(self._get_motif_set(encoded_dataset)) for encoded_dataset in encoded_datasets],
                                   "number_of_tps": [self._compute_tp_sequence_count(encoded_dataset) for encoded_dataset in encoded_datasets]})

        output_file_path = results_folder / "number_of_features_per_subset.tsv"

        results_df.to_csv(output_file_path, index=False, sep="\t")

        return ReportOutput(output_file_path, "Number of significant features found in each data subset")

    def _write_pairwise_intersections(self, encoded_datasets, results_folder):
        result = {"first_subset_idx": [], "second_subset_idx": [],
                  "first_subset_number_of_motifs": [], "second_subset_number_of_motifs": [],
                  "intersecting_motifs": []}

        for first_subset_idx in range(self.n_splits):
            for second_subset_idx in range(first_subset_idx + 1, self.n_splits):
                first_motif_set = self._get_motif_set(encoded_datasets[first_subset_idx])
                second_motif_set = self._get_motif_set(encoded_datasets[second_subset_idx])

                pairwise_intersection = set.intersection(first_motif_set, second_motif_set)

                result["first_subset_idx"].append(first_subset_idx)
                result["first_subset_number_of_motifs"].append(len(first_motif_set))
                result["second_subset_idx"].append(second_subset_idx)
                result["second_subset_number_of_motifs"].append(len(second_motif_set))
                result["intersecting_motifs"].append(len(pairwise_intersection))

                # todo: somehow add the intersection of positive predictions in first and second sets tps_first_subset/tps_second_subset
                # should also the first file info be in here??


        output_file_path = results_folder / "pairwise_intersections.tsv"

        results_df = pd.DataFrame(result)
        results_df.to_csv(output_file_path, index=False, sep="\t")

        return ReportOutput(output_file_path, "Pairwise intersections between the significant features in each data subset")

    def _compute_tp_sequence_count(self, encoded_dataset):
        if encoded_dataset is None:
            return 0
        else:
            positives = np.any(encoded_dataset.encoded_data.examples, axis=1)
            y_true = self._get_y_true(encoded_dataset)
            return sum(np.logical_and(positives, y_true))

    def _get_y_true(self, encoded_dataset):
        label_name = self.label_config.get_labels_by_name()[0]
        label = self.label_config.get_label_object(label_name)

        return np.array([cls == label.positive_class for cls in encoded_dataset.encoded_data.labels[label_name]])


    def _write_multi_intersection_motifs(self, encoded_datasets, results_folder):
        output_file_path = results_folder / "multi_intersection_motifs.tsv"

        motif_sets = [self._get_motif_set(encoded_dataset) for encoded_dataset in encoded_datasets]

        multi_intersection = list(sorted(set.intersection(*motif_sets)))

        motifs = [PositionalMotifHelper.string_to_motif(string, value_sep="&", motif_sep="-") for string in multi_intersection]
        PositionalMotifHelper.write_motifs_to_file(motifs, output_file_path)

        return ReportOutput(output_file_path, f"Intersection of significant motifs across all {self.n_splits} data subsets")

    def _get_label_config(self):
        label_config = LabelHelper.create_label_config([self.label], self.dataset, SignificantMotifOverlap.__name__,
                                                       f"{SignificantMotifOverlap.__name__}/label")
        EncoderHelper.check_positive_class_labels(label_config, f"{SignificantMotifOverlap.__name__}/label")

        return label_config

    def _encode_datasets(self, data_subsets):
        encoded_data_subsets = []

        for i, data_subset in enumerate(data_subsets):
            try:
                encoded_dataset = self._encode_subset(data_subset, i)
            except AssertionError:
                warnings.warn(
                    f"{SignificantMotifOverlap.__name__}: No significant features were found for data subset {i}. "
                    f"Please try decreasing the values for parameters 'min_precision' or 'min_recall' to find more features.")

                encoded_dataset = None

            encoded_data_subsets.append(encoded_dataset)

        return encoded_data_subsets

    # def _get_significant_features_per_subset(self, encoded_datasets):
    #     features_per_subset = []
    #
    #     for encoded_dataset in encoded_datasets:
    #         if encoded_dataset is None:
    #             features_per_subset.append(set())
    #         else:
    #             features_per_subset.append(set(encoded_dataset.encoded_data.feature_names))
    #
    #     return features_per_subset

    def _get_splitted_dataset(self):
        data_subsets = []
        subset_indices = self._get_subset_indices(self.dataset.get_example_count())

        for i in range(self.n_splits):
            data_subset_path = PathBuilder.build(self.result_path / "datasets" / f"split_{i}")
            new_dataset = self.dataset.make_subset(subset_indices[i], data_subset_path, Dataset.SUBSAMPLED)
            new_dataset.name = f"{self.dataset.name}_subset{i}"
            data_subsets.append(new_dataset)

        return data_subsets

    def _get_subset_indices(self, n_examples):
        example_indices = list(range(n_examples))

        random.seed(self.random_seed)
        random.shuffle(example_indices)
        random.seed(None)

        return [example_indices[i::self.n_splits] for i in range(self.n_splits)]

    def _encode_subset(self, data_subset, i):
        logging.info(f"{SignificantMotifOverlap.__name__}: Encoding data subset {i+1}/{self.n_splits}.")

        encoder = SignificantMotifEncoder.build_object(data_subset, **{"max_positions": self.max_positions,
                                                                      "min_precision": self.min_precision,
                                                                      "min_recall": self.min_recall,
                                                                      "min_true_positives": self.min_true_positives,
                                                                      "generalize_motifs": False,
                                                                      "label": None,
                                                                      "name": f"motif_encoder_{data_subset.name}"})

        encoder_params = EncoderParams(result_path=self.result_path / "encoded_data" / f"split_{i}",
                                       pool_size=self.number_of_processes,
                                       learn_model=True,
                                       label_config=self.label_config) # todo need individual label config per data subset???

        encoded_dataset = encoder.encode(data_subset, encoder_params)

        logging.info(f"{SignificantMotifOverlap.__name__}: Finished encoding data subset {i+1}/{self.n_splits}.")

        return encoded_dataset

