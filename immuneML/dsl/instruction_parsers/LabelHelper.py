import warnings

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.ParameterValidator import ParameterValidator


class LabelHelper:

    @staticmethod
    def check_label_format(labels: list, instruction_name: str, yaml_location: str):
        ParameterValidator.assert_type_and_value(labels, list, instruction_name, f'{yaml_location}/labels')

        assert all(isinstance(label, str) or isinstance(label, dict) for label in labels), \
            f"{instruction_name}: labels under {yaml_location} were not defined properly. The list of labels has to either be a list of " \
            f"label names, or there can be a parameter 'positive_class' defined under the label name, for example:\n" \
            f"labels: # one label with no positive class (T1D) and one with a positive class (CMV)\n" \
            f"- T1D\n" \
            f"- CMV: # when defining a positive class, make sure to use the correct indentation\n" \
            f"    positive_class: True\n" \

        assert all(len(list(label.keys())) == 1 and isinstance(list(label.values())[0], dict) and 'positive_class' in list(label.values())[0]
                   and len(list(list(label.values())[0].keys())) == 1 for label in [l for l in labels if isinstance(l, dict)]), \
            f"{instruction_name}: The only legal parameter under a label name is 'positive_class'. If 'positive_class' is not specified, please " \
            f"remove the colon after the label name. "

    @staticmethod
    def create_label_config(labels: list, dataset: Dataset, instruction_name: str, yaml_location: str) -> LabelConfiguration:
        LabelHelper.check_label_format(labels, instruction_name, yaml_location)

        label_config = LabelConfiguration()
        for label in labels:
            label_name = label if isinstance(label, str) else list(label.keys())[0]
            positive_class = label[label_name]['positive_class'] if isinstance(label, dict) else None
            if dataset.labels is not None and label_name in dataset.labels:
                label_values = list(dataset.labels[label_name])
            elif hasattr(dataset, "get_metadata"):
                label_values = list(set(dataset.get_metadata([label_name])[label_name]))
            else:
                label_values = []
                warnings.warn(f"{instruction_name}: for {yaml_location}, label values could not be recovered for label "
                              f"{label}, using empty list instead. This issue may occur due to improper loading of dataset {dataset.name},"
                              f"and could cause problems with some encodings.")

            label_config.add_label(label_name, label_values, positive_class=positive_class)

        return label_config
