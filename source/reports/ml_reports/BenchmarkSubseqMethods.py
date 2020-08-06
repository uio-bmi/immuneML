import warnings

from scripts.specification_util import update_docs_per_mapping
from source.ml_methods.DeepRC import DeepRC
from source.reports.ml_reports.BenchmarkHPSettings import BenchmarkHPSettings
from source.util.ParameterValidator import ParameterValidator
from source.visualization.ErrorBarMeaning import ErrorBarMeaning


class BenchmarkSubseqMethods(BenchmarkHPSettings):
    """
    Report for HyperParameterOptimization: Similar to :py:obj:`~source.reports.ml_reports.BenchmarkHPSettings.BenchmarkHPSettings`,
    this report plots the performance of certain combinations of encodings and ML methods.

    Similarly to BenchmarkHPSettings, the performances are grouped by label (horizontal panels).
    However, the bar color is determined by the ml method class (thus several ML methods with different parameters
    may be grouped together) and the vertical panel grouping is determined by the subsequence size used for motif recovery.
    This subsequence size is either the k-mer size or the kernel size (DeepRC).

    This report can only be used to plot the results for setting combinations using k-mer encoding with continuous k-mers
    (in combination with any ML method), or DeepRC encoding + ml method.

    This report can only be used with HPOptimization instruction under assessment/reports/hyperparameter.

    Attributes:

        errorbar_meaning (:py:obj:`~source.visualization.ErrorBarMeaning.ErrorBarMeaning`): The value that
            the error bar should represent. For options see :py:obj:`~source.visualization.ErrorBarMeaning.ErrorBarMeaning`.


    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_hp_report:
            BenchmarkHPSettings:
                errorbar_meaning: STANDARD_ERROR

    """

    @classmethod
    def build_object(cls, **kwargs):
        valid_values = [item.name.lower() for item in ErrorBarMeaning]
        ParameterValidator.assert_in_valid_list(kwargs["errorbar_meaning"], valid_values, "BenchmarkSubseqMethods", "errorbar_meaning")
        errorbar_meaning = ErrorBarMeaning[kwargs["errorbar_meaning"].upper()]
        return BenchmarkSubseqMethods(errorbar_meaning, kwargs["name"] if "name" in kwargs else None)

    def __init__(self, errorbar_meaning: ErrorBarMeaning, name: str = None):
        super(BenchmarkSubseqMethods, self).__init__(errorbar_meaning, name)
        self.vertical_grouping = "subsequence_size"
        self.result_name = "benchmark_subseq_result"


    def _get_vertical_grouping(self, assessment_item):
        subseq_size = "N/A"

        if isinstance(assessment_item.hp_setting.ml_method, DeepRC):
            subseq_size = assessment_item.hp_setting.ml_method.kernel_size
        elif assessment_item.hp_setting.encoder.__module__.endswith("KmerFrequencyEncoder"):
            subseq_size = assessment_item.hp_setting.encoder_params['k']

        return f"k-mer/kernel\nsize {subseq_size}"


    def _get_color_grouping(self, assessment_item):
        return assessment_item.hp_setting.ml_method.__class__.__name__


    @staticmethod
    def get_documentation():
        doc = str(BenchmarkSubseqMethods.__doc__)
        valid_values = str([option.name for option in ErrorBarMeaning])[1:-1].replace("'", "`")
        mapping = {
            "For options see :py:obj:`~source.visualization.ErrorBarMeaning.ErrorBarMeaning`.": f"Valid values are: {valid_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc

    def _check_valid_assessment_item(self, assessment_item):
        is_valid = False

        if isinstance(assessment_item.hp_setting.ml_method, DeepRC):
            is_valid = True
        elif assessment_item.hp_setting.encoder.__module__.endswith("KmerFrequencyEncoder"):
            if assessment_item.hp_setting.encoder_params['sequence_encoding'].lower() in ('continuous_kmer', 'imgt_gapped_kmer'):
                is_valid = True

        return is_valid


    def check_prerequisites(self):
        run_report = True

        if not hasattr(self, "hp_optimization_state") or self.hp_optimization_state is None:
            warnings.warn(f"{self.__class__.__name__} can only be executed as a hyperparameter report. BenchmarkSubseqMethods report will not be created.")
            run_report = False

        if not hasattr(self, "result_path") or self.result_path is None:
            warnings.warn(f"{self.__class__.__name__} requires an output 'path' to be set. BenchmarkSubseqMethods report will not be created.")
            run_report = False

        for assessment_state in self.hp_optimization_state.assessment_states:
            for label_state in assessment_state.label_states.values():
                for assessment_item in label_state.assessment_items.values():
                    if not self._check_valid_assessment_item(assessment_item):
                        warnings.warn(f"{self.__class__.__name__} can only be used on encoder-ML method combinations that use k-mer encoding"
                                      f"with continuous k-mers, or DeepRC. BenchmarkSubseqMethods report will not be created.")
                        run_report = False

        return run_report
