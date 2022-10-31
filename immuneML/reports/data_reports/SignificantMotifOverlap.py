import random
from pathlib import Path
import pandas as pd
import warnings
import logging

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
        features_per_subset = self._get_significant_features_per_subset()
        result_files = self._write_result_files(features_per_subset)

        return ReportResult(name=self.name,
                            info=f"Analysis of significant motifs found across {len(features_per_subset)} subsets of the dataset.",
                            output_tables=result_files)


        # todo many-set overlap?
            # export all raw lists of amino acids
            # number of overlapping between each of them...
            # plot pairwise union sizes, see when it starts to go down?? -> multiple cutoffs... but we want to use one cutoff and multiple weighting params not multiple cutoffs
            # to see that with shitty parameterization there is no overlap, and better regularisation is more overlap
            # but we also find more motifs?? generally:number overlapping is good enough. also finding more motifs in all cases and having more overlapping is good.. right?
            # no also need to show total number found -> just in case you just find 100x more, but not more overlapping.. that would be bad. just as double check

    def _write_result_files(self, features_per_subset):
        results_folder = PathBuilder.build(self.result_path / "feature_intersections")


        subset_sizes_output = self._write_subset_sizes(features_per_subset, results_folder)
        pairwise_intersection_output = self._write_pairwise_intersections(features_per_subset, results_folder)
        multi_intersection_motifs = self._write_multi_intersection_motifs(features_per_subset, results_folder)

        return [subset_sizes_output, pairwise_intersection_output, multi_intersection_motifs]

    def _write_subset_sizes(self, features_per_subset, results_folder):
        results_df = pd.DataFrame({"subset_idx": list(range(len(features_per_subset))),
                                   "number_of_features": [len(feature_list) for feature_list in features_per_subset]})

        output_file_path = results_folder / "number_of_features_per_subset.tsv"

        results_df.to_csv(output_file_path, index=False, sep="\t")

        return ReportOutput(output_file_path, "Number of significant features found in each data subset")

    def _write_pairwise_intersections(self, features_per_subset, results_folder):
        result = {"first_subset_idx": [], "second_subset_idx": [],
                  "first_subset_size": [], "second_subset_size": [],
                  "intersection_size": []}

        for first_subset_idx in range(len(features_per_subset)):
            for second_subset_idx in range(first_subset_idx + 1, len(features_per_subset)):
                pairwise_intersection = set.intersection(features_per_subset[first_subset_idx],
                                                         features_per_subset[second_subset_idx])

                result["first_subset_idx"].append(first_subset_idx)
                result["first_subset_size"].append(len(features_per_subset[first_subset_idx]))
                result["second_subset_idx"].append(second_subset_idx)
                result["second_subset_size"].append(len(features_per_subset[second_subset_idx]))
                result["intersection_size"].append(len(pairwise_intersection))

        output_file_path = results_folder / "pairwise_intersections.tsv"

        results_df = pd.DataFrame(result)
        results_df.to_csv(output_file_path, index=False, sep="\t")

        return ReportOutput(output_file_path, "Pairwise intersections between the significant features in each data subset")

    def _write_multi_intersection_motifs(self, features_per_subset, results_folder):
        output_file_path = results_folder / "multi_intersection_motifs.tsv"

        multi_intersection = list(sorted(set.intersection(*features_per_subset)))

        motifs = [PositionalMotifHelper.string_to_motif(string, value_sep="&", motif_sep="-") for string in multi_intersection]
        PositionalMotifHelper.write_motifs_to_file(motifs, output_file_path)

        return ReportOutput(output_file_path, f"Intersection of significant motifs across all {len(features_per_subset)} data subsets")

    def _get_label_config(self):
        label_config = LabelHelper.create_label_config([self.label], self.dataset, SignificantMotifOverlap.__name__,
                                                       f"{SignificantMotifOverlap.__name__}/label")
        EncoderHelper.check_positive_class_labels(label_config, f"{SignificantMotifOverlap.__name__}/label")

        return label_config

    def _get_significant_features_per_subset(self):
        features_per_subset = []

        for i, data_subset in enumerate(self._split_dataset()):
            try:
                encoded_dataset = self._encode_subset(data_subset, i)
                features_per_subset.append(set(encoded_dataset.encoded_data.feature_names))
            except AssertionError:
                warnings.warn(f"{SignificantMotifOverlap.__name__}: No significant features were found for data subset {i}. "
                              f"Please try decreasing the values for parameters 'min_precision' or 'min_recall' to find more features.")
                features_per_subset.append(set())

        return features_per_subset

    def _split_dataset(self):
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
                                                       label_config=self.label_config)

        encoded_dataset = encoder.encode(data_subset, encoder_params)

        logging.info(f"{SignificantMotifOverlap.__name__}: Finished encoding data subset {i+1}/{self.n_splits}.")

        return encoded_dataset

