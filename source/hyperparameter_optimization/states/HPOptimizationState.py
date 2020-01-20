from source.environment.LabelConfiguration import LabelConfiguration
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.config.SplitConfig import SplitConfig
from source.hyperparameter_optimization.strategy.HPOptimizationStrategy import HPOptimizationStrategy


class HPOptimizationState:

    def __init__(self, dataset, hp_strategy: HPOptimizationStrategy, hp_settings: list,
                 assessment: SplitConfig, selection: SplitConfig, metrics: set,
                 label_configuration: LabelConfiguration, path: str = None, context: dict = None, batch_size: int = 10):

        # initial attributes
        self.dataset = dataset
        self.selection_config = selection
        self.hp_strategy = hp_strategy
        assert all(isinstance(hp_setting, HPSetting) for hp_setting in hp_settings), \
            "HPOptimizationState: object of other type passed in instead of HPSetting."
        self.hp_settings = hp_settings
        self.path = path
        self.batch_size = batch_size
        self.label_configuration = label_configuration
        self.metrics = metrics
        self.assessment_config = assessment
        self.context = context

        # computed attributes
        self.assessment_states = []
