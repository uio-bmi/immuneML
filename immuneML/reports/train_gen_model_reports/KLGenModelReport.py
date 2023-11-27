from functools import partial
from pathlib import Path
import bionumpy as bnp
from .TrainGenModelReport import TrainGenModelReport
from ..ReportOutput import ReportOutput
from ..ReportResult import ReportResult
from ...data_model.dataset.Dataset import Dataset
from ...ml_methods.generative_models.GenerativeModel import GenerativeModel
from ...ml_methods.generative_models.KLEvaluator import evaluate_similarities, KLEvaluator
from ...ml_methods.generative_models.MultinomialKmerModel import estimate_kmer_model
from ...util.PathBuilder import PathBuilder


class KLGenModelReport(TrainGenModelReport):
    """
    TrainGenModel reports show some type of features or statistics comparing two datasets: the original and generated
    one, potentially in combination with the trained model. These reports can only be used inside TrainGenModel
    instruction with the aim of comparing two datasets: the dataset used to train a generative model and the dataset
    created from the trained model.
    """

    def __init__(self, original_dataset: Dataset = None, generated_dataset: Dataset = None, result_path: Path = None,
                 name: str = None, number_of_processes: int = 1, model: GenerativeModel = None):
        """
        The arguments defined below are set at runtime by the instruction.
        Concrete classes inheriting DataComparisonReport may include additional parameters that will be set by the user
        in the form of input arguments (e.g., from the YAML file).

        Args:

            original_dataset (Dataset): a dataset object (can be repertoire, receptor or sequence dataset, depending
            on the specific report) provided as input to the TrainGenModel instruction

            generated_dataset (Dataset): a dataset object as produced from the generative model after being trained on
            the original dataset

            result_path (Path): location where the results (plots, tables, etc.) will be stored

            name (str): user-defined name of the report used in the HTML overview automatically generated by the
            platform from the key used to define the report in the YAML

            number_of_processes (int): how many processes should be created at once to speed up the analysis.
            For personal machines, 4 or 8 is usually a good choice.

            model (GenerativeModel): trained generative model from the instruction

        """
        super().__init__(name=name, number_of_processes=number_of_processes)
        self.original_dataset = original_dataset
        self.generated_dataset = generated_dataset
        self.model = model
        self.result_path = result_path

    @staticmethod
    def get_title():
        return "KLTrainGenModel reports"



    @classmethod
    def build_object(cls, **kwargs):
        """
        Creates the object of the subclass of the Report class from the parameters so that it can be used in the analysis. Depending on the type of
        the report, the parameters provided here will be provided in parsing time, while the other necessary parameters (e.g., subset of the data from
        which the report should be created) will be provided at runtime. For more details, see specific direct subclasses of this class, describing
        different types of reports.

        Args:

            **kwargs: keyword arguments that will be provided by users in the specification (if immuneML is used as a command line tool) or in the
             dictionary when calling the method from the code, and which should be used to create the report object

        Returns:

            the object of the appropriate report class

        """
        location = cls.__name__
        return cls(**kwargs)
        # ParameterValidator.assert_type_and_value(kwargs["imgt_positions"], bool, location, "imgt_positions")
        # ParameterValidator.assert_type_and_value(kwargs["relative_frequency"], bool, location, "relative_frequency")
        # ParameterValidator.assert_type_and_value(kwargs["split_by_label"], bool, location, "split_by_label")
        #
        # if kwargs["label"] is not None:
        #     ParameterValidator.assert_type_and_value(kwargs["label"], str, location, "label")
        #
        #     if kwargs["split_by_label"] is False:
        #         warnings.warn(f"{location}: label is set but split_by_label was False, setting split_by_label to True")
        #         kwargs["split_by_label"] = True
        #
        # return AminoAcidFrequencyDistribution(**kwargs)
        #
        # pass

    def _compute_kl_divergence(self):
        """
        Computes the KL divergence between the original and generated dataset

        Returns:
            KL divergence value
        """
        return self._get_kmer_kl_evaluator()


    # def _get_kmer_model(self, sequences):
    #     sequences = data_set.get_attribute("sequence_aa")
    #     kmers = bnp.sequence.get_kmers(sequences, 3)
    #     model = estimate_kmer_model(kmers)
    #     return model

    def _get_kmer_kl_evaluator(self):
        o_kmers, g_kmers = (bnp.sequence.get_kmers(dataset.get_attribute("sequence_aa"), 3)
                            for dataset in (self.original_dataset, self.generated_dataset))
        estimator = partial(estimate_kmer_model, prior_count=1.0)
        return KLEvaluator(o_kmers, g_kmers, estimator)
        # kls = evaluate_similarities(o_kmers, g_kmers, estimator)
        # return kls

    def _get_transitions_kl(self):
        o_kmers, g_kmers = (bnp.sequence.get_kmers(dataset.get_attribute("sequence_aa"), 3)
                            for dataset in (self.original_dataset, self.generated_dataset))
        estimator = partial(estimate_kmer_model, prior_count=1.0)

    def _generate(self) -> ReportResult:
        """
        The function that needs to be implemented by the Report subclasses which actually creates the report (figures, tables, text files), depending
        on the specific aim of the report. After checking all prerequisites (e.g., if all parameters were set properly), generate_report() will call
        this function and return its result.

        Returns:

            ReportResult object which encapsulates all outputs (figure, table, and text files) so that they can be conveniently linked to in the
            final output of instructions

        """
        PathBuilder.build(self.result_path)
        evaluator = self._get_kmer_kl_evaluator()
        tables = [self._write_output_table(evaluator.get_worst_true_sequences(),
                                           self.result_path / "worst_true_sequences.tsv",
                                           name="Original sequences that don't fit the generated model"),
                  self._write_output_table(evaluator.get_worst_simulated_sequences(),
                                           self.result_path / "worst_simulated_sequences.tsv",
                                           name="Generated sequences that don't fit with the original model")]
        # tables = []
        figures = []
        #
        # tables.append(self._write_output_table(freq_dist,
        #                                        self.result_path / "amino_acid_frequency_distribution.tsv",
        #                                        name="Table of amino acid frequencies"))
        #
        figures.append(self._plot_simulated(evaluator=evaluator))
        figures.append(self._plot_original(evaluator=evaluator))
        # self._safe_plot(plot_callable="_plot_distribution"))
        #
        # if self.split_by_label:
        #     frequency_change = self._compute_frequency_change(freq_dist)
        #
        #     tables.append(self._write_output_table(frequency_change,
        #                                            self.result_path / f"frequency_change.tsv",
        #                                            name=f"Frequency change between classes"))
        #     figures.append(self._safe_plot(frequency_change=frequency_change, plot_callable="_plot_frequency_change"))

        info_text = '''Estimated KL divergence between the kmer distributions in the original and generated datasets. Toghether with the sequences that contribute the most to the divergence.
        KL(original || generated) = {:.2f},  KL(generated || original) = {:.2f}'''.format(evaluator.true_kl(), evaluator.simulated_kl())
        return ReportResult(name=self.name,
                            info=info_text,
                            output_figures=figures,
                            output_tables=[table for table in tables if table is not None])

    def _plot_simulated(self, evaluator: KLEvaluator):
        file_path = self.result_path / "bad_simulated_sequences.html"
        figure = evaluator.simulated_plot()
        figure.write_html(str(file_path))

        return ReportOutput(path=file_path, name="Generated Sequences that contributes most to the KL divergence")

    def _plot_original(self, evaluator: KLEvaluator):
        file_path = self.result_path / "bad_original_sequences.html"
        figure = evaluator.original_plot()
        figure.write_html(str(file_path))

        return ReportOutput(path=file_path, name="Original Sequences that contributes most to the KL divergence")