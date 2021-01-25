import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.LogisticRegression import LogisticRegression
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.ml_methods.RandomForestClassifier import RandomForestClassifier
from immuneML.ml_methods.SVM import SVM
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class MotifSeedRecovery(MLReport):
    """
    This report can be used to show how well implanted motifs (for example, through the Simulation instruction) can
    be recovered by various machine learning methods using the k-mer encoding.
    This report creates a boxplot, where the x axis (box grouping) represents the maximum possible overlap between
    an implanted motif seed and a kmer feature (measured in number of positions), and the y axis shows the coefficient size
    of the respective kmer feature. If the machine learning method has learned the implanted motif seeds, the coefficient
    size is expected to be largest for the kmer features with high overlap to the motif seeds.

    Note that to use this report, the following criteria must be met:
    - KmerFrequencyEncoder must be used.
    - For each label, the implanted motif seeds relevant to that label must be specified


    To find the overlap score between kmer features and implanted motif seeds, the two sequences are compared in a sliding
    window approach, and the maximum overlap is calculated.

    Overlap scores between kmer features and implanted motifs are calculated differently based on hamming distance was
    allowed during implanting.

    .. indent with spaces
    .. code-block::

        Without hamming distance:
        Seed:     AAA  -> score = 3
        Feature: xAAAx
                  ^^^

        Seed:     AAA  -> score = 0
        Feature: xAAxx

        With hamming distance:
        Seed:     AAA  -> score = 3
        Feature: xAAAx
                  ^^^

        Seed:     AAA  -> score = 2
        Feature: xAAxx
                  ^^

        Furthermore, gap positions in the motif seed are ignored:
        Seed:     A/AA  -> score = 3
        Feature: xAxAAx
                  ^/^^


    Arguments:
        implanted_motifs_per_label (dict): a nested dictionary that specifies the motif seeds that were implanted in
            the given dataset. The first level of keys in this dictionary represents the different labels. In the
            inner dictionary there should be two keys: "seeds" and "hamming_distance"
                seeds: a list of motif seeds. The seeds may contain gaps, specified by a '/' symbol.
                hamming_distance: A boolean value that specifies whether hamming distance was allowed when implanting the
                    motif seeds for a given label. Note that this applies to all seeds for this label.
                gap_sizes: a list of all the possible gap sizes that were used when implanting a gapped motif seed.
                    When no gapped seeds are used, this value has no effect.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_motif_report:
            MotifSeedRecovery:
                implanted_motifs_per_label:
                    CD:
                        seeds:
                        - AA/A
                        - AAA
                        hamming_distance: False
                        gap_sizes:
                        - 0
                        - 1
                        - 2
                    T1D
                        seeds:
                        - CC/C
                        - CCC
                        hamming_distance: True
                        gap_sizes:
                        - 2


    """

    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_keys_present(kwargs.keys(),
                                               ["implanted_motifs_per_label"],
                                               "MotifSeedRecovery", "MotifSeedRecovery report")

        implanted_motifs_per_label = kwargs["implanted_motifs_per_label"]

        ParameterValidator.assert_type_and_value(implanted_motifs_per_label, dict,
                                                 "MotifSeedRecovery",
                                                 f"implanted_motifs_per_label")

        for label in implanted_motifs_per_label.keys():
            ParameterValidator.assert_type_and_value(implanted_motifs_per_label[label], dict,
                                                     "MotifSeedRecovery",
                                                     f"implanted_motifs_per_label/{label}")

            ParameterValidator.assert_keys_present(implanted_motifs_per_label[label].keys(),
                                                   ["hamming_distance", "seeds", "gap_sizes"],
                                                   "MotifSeedRecovery", f"implanted_motifs_per_label/{label}")
            ParameterValidator.assert_type_and_value(implanted_motifs_per_label[label]["hamming_distance"], bool,
                                                     "MotifSeedRecovery", f"implanted_motifs_per_label/{label}/hamming_distance")
            ParameterValidator.assert_type_and_value(implanted_motifs_per_label[label]["gap_sizes"], list,
                                                     "MotifSeedRecovery",
                                                     f"implanted_motifs_per_label/{label}/gap_sizes")
            ParameterValidator.assert_type_and_value(implanted_motifs_per_label[label]["seeds"], list,
                                                     "MotifSeedRecovery", f"implanted_motifs_per_label/{label}/seeds")
            for gap_size in implanted_motifs_per_label[label]["gap_sizes"]:
                ParameterValidator.assert_type_and_value(gap_size, int, "MotifSeedRecovery",
                                                         f"implanted_motifs_per_label/{label}/gap_sizes", min_inclusive=0)
            for seed in implanted_motifs_per_label[label]["seeds"]:
                ParameterValidator.assert_type_and_value(seed, str, "MotifSeedRecovery",
                                                         f"implanted_motifs_per_label/{label}/seeds")

        return MotifSeedRecovery(implanted_motifs_per_label)

    def __init__(self, implanted_motifs_per_label, train_dataset: Dataset = None,
                 test_dataset: Dataset = None, method: MLMethod = None, result_path: Path = None, name: str = None):
        super().__init__(train_dataset, test_dataset, method, result_path, name)
        self.implanted_motifs_per_label = implanted_motifs_per_label
        self.label = None
        self._param_field = None
        self._y_axis_title = None
        self._x_axis_title = None

    def _generate(self):
        PathBuilder.build(self.result_path)

        self._set_plotting_parameters()

        plot_df = self._retrieve_plot_data()
        report_output_table = self._write_results_table(plot_df)
        report_output_fig = self._plot(plot_df, "motif_seed_recovery")

        return ReportResult(self.name, output_tables=[report_output_table],
                            output_figures=[report_output_fig])

    def _write_results_table(self, plotting_data):
        filepath = self.result_path / "motif_seed_recovery.csv"
        plotting_data.to_csv(filepath, index=False)
        return ReportOutput(path=filepath, name="motif seed recovery csv")

    def _set_plotting_parameters(self):
        if isinstance(self.method, RandomForestClassifier):
            self._param_field = "feature_importances"
            self._y_axis_title = "Feature importance"
        else:
            # SVM, logistic regression, ...
            self._param_field = "coefficients"
            self._y_axis_title = "Coefficient value"

        if self.implanted_motifs_per_label[self.label]["hamming_distance"]:
            self._x_axis_title = "Positions overlap between feature and motif seeds<br>(hamming distance allowed)"
        else:
            self._x_axis_title = "Positions overlap between feature and motif seeds"

    def _retrieve_plot_data(self):
        seeds = self._get_implanted_seeds()
        overlap_fn = self._get_overlap_fn()
        features = self._retrieve_feature_names()

        plot_df = self.calculate_seed_overlap(seeds, features, overlap_fn)
        plot_df["coefficients"] = self.method.get_params(self.label)[self._param_field]
        return plot_df

    def _get_implanted_seeds(self):
        return self.implanted_motifs_per_label[self.label]["seeds"]

    def _get_overlap_fn(self):
        is_hamming_distance = self.implanted_motifs_per_label[self.label]["hamming_distance"]
        overlap_fn = self.hamming_overlap if is_hamming_distance else self.identical_overlap
        return overlap_fn

    def _retrieve_feature_names(self):
        if self.train_dataset and self.train_dataset.encoded_data:
            return self.train_dataset.encoded_data.feature_names

    def _plot(self, plotting_data, output_name):
        if plotting_data.empty:
            logging.warning(f"Coefficients: empty data subset specified, skipping {output_name} plot...")
        else:

            filename = self.result_path / f"{output_name}.html"

            import plotly.express as px
            figure = px.box(plotting_data, x="max_seed_overlap", y="coefficients", labels={
                "max_seed_overlap": self._x_axis_title,
                "coefficients": self._y_axis_title
            }, template='plotly_white',
                         color_discrete_sequence=px.colors.diverging.Tealrose)
            # figure.update_layout(title={"text":self.title, "x":0.5, "font": {"size":14}})

            figure.write_html(str(filename))

            return ReportOutput(filename, f"Overlap between implanted motif seeds and features versus {self._y_axis_title.lower()}")

    def hamming_overlap(self, seed, feature):
        return sum(np.array(list(seed)) == np.array(list(feature)))

    def identical_overlap(self, seed, feature):
        if "/" in seed:
            exclude_idx = seed.index("/")
            seed = seed[:exclude_idx] + seed[exclude_idx + 1:]
            feature = feature[:exclude_idx] + feature[exclude_idx + 1:]

        while feature.startswith("-"):
            feature = feature[1:]
            seed = seed[1:]

        while feature.endswith("-"):
            feature = feature[:-1]
            seed = seed[:-1]

        return int(seed == feature) * len(seed)


    def identical_overlap(self, seed, feature):
        if "/" in seed:
            exclude_idx_start = seed.index("/")
            exclude_idx_end = seed.rindex("/")
            seed = seed[:exclude_idx_start] + seed[exclude_idx_end + 1:]
            feature = feature[:exclude_idx_start] + feature[exclude_idx_end + 1:]

        while feature.startswith("-"):
            feature = feature[1:]
            seed = seed[1:]

        while feature.endswith("-"):
            feature = feature[:-1]
            seed = seed[:-1]

        return int(seed == feature) * len(seed)

    def max_overlap_sliding(self, seed, feature, overlap_fn):
        max_score = 0

        sizes = self.implanted_motifs_per_label[self.label]["gap_sizes"]

        for gap_size in sizes:
            gap_adjusted_seed = seed.replace("/", "/" * gap_size)

            padding = "-" * (len(gap_adjusted_seed) - 1)

            padded_feature = padding + feature + padding

            for start_idx in range(0, len(feature) + len(padding)):
                feature_slice = padded_feature[start_idx:start_idx + len(gap_adjusted_seed)]
                max_score = max(max_score, overlap_fn(gap_adjusted_seed, feature_slice))

        return max_score

    def calculate_seed_overlap(self, motif_seeds, features, overlap_fn):
        seed_df = pd.DataFrame({"features": features})

        for seed in motif_seeds:
            seed_df[seed] = [self.max_overlap_sliding(seed, feature, overlap_fn) for feature in seed_df["features"]]

        seed_df["max_seed_overlap"] = seed_df.drop("features", axis=1).max(axis=1)
        seed_df = seed_df[["features", "max_seed_overlap"]]

        return seed_df

    def check_prerequisites(self):
        location = "MotifSeedRecovery"

        run_report = True

        if not any([isinstance(self.method, legal_method) for legal_method in (RandomForestClassifier, LogisticRegression, SVM)]):
            logging.warning(f"{location} report can only be created for RandomForestClassifier, LogisticRegression or SVM, but got "
                            f"{type(self.method).__name__} instead. {location} report will not be created.")
            run_report = False

        if self.label not in self.implanted_motifs_per_label.keys():
            warnings.warn(
                f"{location}: no implanted motifs were specified for the label '{self.label}'. "
                f"These motifs should be specified under 'implanted_motifs_per_label'. {location} report will not be created.")
            run_report = False

        if self.train_dataset.encoded_data is None or self.train_dataset.encoded_data.examples is None or self.train_dataset.encoded_data.feature_names is None:
            warnings.warn(
                f"{location}: this report can only be created for an encoded dataset with specified feature names. {location} report will not be created.")
            run_report = False

        if self.train_dataset.encoded_data.encoding != "KmerFrequencyEncoder":
            warnings.warn(
                f"{location}: this report can only be created for a dataset encoded with the KmerFrequencyEncoder. {location} report will not be created.")
            run_report = False

        return run_report
