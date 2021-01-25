from pathlib import Path

from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.hyperparameter_optimization.states.HPLabelState import HPLabelState


class HPAssessmentState:

    def __init__(self, split_index: int, train_val_dataset, test_dataset, path: Path, label_configuration: LabelConfiguration):
        self.split_index = split_index
        self.train_val_dataset = train_val_dataset
        self.test_dataset = test_dataset
        self.path = path
        self.train_val_data_reports = []
        self.test_data_reports = []

        # computed
        self.label_states = {label: HPLabelState(label, label_configuration.get_auxiliary_labels(label))
                             for label in label_configuration.get_labels_by_name()}
