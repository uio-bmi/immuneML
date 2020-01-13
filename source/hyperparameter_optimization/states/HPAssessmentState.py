from source.environment.LabelConfiguration import LabelConfiguration
from source.hyperparameter_optimization.states.HPLabelState import HPLabelState


class HPAssessmentState:

    def __init__(self, split_index: int, train_val_dataset, test_dataset, path: str, label_configuration: LabelConfiguration):

        self.split_index = split_index
        self.train_val_dataset = train_val_dataset
        self.test_dataset = test_dataset
        self.path = path

        # computed
        self.label_states = {label: HPLabelState(label, label_configuration.get_auxiliary_labels(label))
                             for label in label_configuration.get_labels_by_name()}
