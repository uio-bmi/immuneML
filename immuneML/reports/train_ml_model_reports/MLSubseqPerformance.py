import warnings

from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.ml_methods.DeepRC import DeepRC
from immuneML.reports.train_ml_model_reports.MLSettingsPerformance import MLSettingsPerformance


class MLSubseqPerformance(MLSettingsPerformance):
    """
    Report for TrainMLModel: Similar to :py:obj:`~immuneML.reports.ml_reports.MLSettingsPerformance.MLSettingsPerformance`,
    this report plots the performance of certain combinations of encodings and ML methods.

    Similarly to MLSettingsPerformance, the performances are grouped by label (horizontal panels).
    However, the bar color is determined by the ml method class (thus several ML methods with different parameters
    may be grouped together) and the vertical panel grouping is determined by the subsequence size used for motif recovery.
    This subsequence size is either the k-mer size or the kernel size (DeepRC).

    This report can only be used to plot the results for setting combinations using k-mer encoding with continuous k-mers
    (in combination with any ML method), or DeepRC encoding + ml method.

    This report can only be used with TrainMLModel instruction under 'reports'.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_hp_report: MLSubseqPerformance

    """

    @classmethod
    def build_object(cls, **kwargs):
        return MLSubseqPerformance(kwargs["name"] if "name" in kwargs else None)

    def __init__(self, name: str = None):
        super().__init__(name)
        self.vertical_grouping = "subsequence_size"
        self.result_name = "subseq_performance"
        self.state = None

    def _get_vertical_grouping(self, assessment_item):
        subseq_size = "N/A"

        if isinstance(assessment_item.hp_setting.ml_method, DeepRC):
            subseq_size = assessment_item.hp_setting.ml_method.kernel_size
        elif assessment_item.hp_setting.encoder.__module__.endswith("KmerFrequencyEncoder"):
            subseq_size = assessment_item.hp_setting.encoder_params['k']

        return f"k-mer/kernel\nsize {subseq_size}"

    def _get_color_grouping(self, assessment_item):
        return assessment_item.hp_setting.ml_method.__class__.__name__

    def _check_valid_assessment_item(self, assessment_item):
        is_valid = False

        if isinstance(assessment_item.hp_setting.ml_method, DeepRC):
            is_valid = True
        elif assessment_item.hp_setting.encoder.__class__.__name__ in KmerFrequencyEncoder.dataset_mapping.values():
            if assessment_item.hp_setting.encoder_params['sequence_encoding'].lower() in ('continuous_kmer', 'imgt_continuous_kmer'):
                is_valid = True

        return is_valid

    def check_prerequisites(self):
        run_report = True

        if self.state is None:
            warnings.warn(
                f"{self.__class__.__name__} can only be executed as a hyperparameter report. MLSubseqPerformance report will not be created.")
            run_report = False

        if self.result_path is None:
            warnings.warn(f"{self.__class__.__name__} requires an output 'path' to be set. {self.__class__.__name__} report will not be created.")
            run_report = False

        for assessment_state in self.state.assessment_states:
            for label_state in assessment_state.label_states.values():
                for assessment_item in label_state.assessment_items.values():
                    if not self._check_valid_assessment_item(assessment_item):
                        warnings.warn(f"{self.__class__.__name__} can only be used on encoder-ML method combinations that use k-mer encoding"
                                      f"with continuous k-mers, or DeepRC. {self.__class__.__name__} report will not be created.")
                        run_report = False

        return run_report
