import copy
import pickle
import warnings

from immuneML.IO.dataset_export.ImmuneMLExporter import ImmuneMLExporter
from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.ElementDataset import ElementDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from immuneML.util.PathBuilder import PathBuilder


class EncoderHelper:

    @staticmethod
    def prepare_training_ids(dataset: Dataset, params: EncoderParams):
        PathBuilder.build(params.result_path)
        if params.learn_model:
            training_ids = dataset.get_example_ids()
            training_ids_path = params.result_path / "training_ids.pickle"
            with training_ids_path.open("wb") as file:
                pickle.dump(training_ids, file)
        else:
            training_ids_path = params.result_path / "training_ids.pickle"
            with training_ids_path.open("rb") as file:
                training_ids = pickle.load(file)
        return training_ids

    @staticmethod
    def get_current_dataset(dataset, context):
        '''Retrieves the full dataset (training+validation+test) if present in context, otherwise return the given dataset'''
        return dataset if context is None or "dataset" not in context else context["dataset"]

    @staticmethod
    def build_comparison_params(dataset, comparison_attributes) -> tuple:
        return (("dataset_identifier", dataset.identifier),
                ("comparison_attributes", tuple(comparison_attributes)),
                ("repertoire_ids", tuple(dataset.get_repertoire_ids())))

    @staticmethod
    def build_comparison_data(dataset: RepertoireDataset, params: EncoderParams,
                              comparison_attributes, sequence_batch_size):

        comp_data = ComparisonData(dataset.get_repertoire_ids(), comparison_attributes,
                                   sequence_batch_size, params.result_path)

        comp_data.process_dataset(dataset)

        return comp_data

    @staticmethod
    def store(encoded_dataset, params: EncoderParams):
        ImmuneMLExporter.export(encoded_dataset, params.result_path)

    @staticmethod
    def sync_encoder_with_cache(cache_params: tuple, encoder_memo_func, encoder, param_names):
        encoder_cache_params = tuple((key, val) for key, val in dict(cache_params).items() if key != 'learn_model')
        encoder_cache_params = (encoder_cache_params, "encoder")

        encoder_from_cache = CacheHandler.memo_by_params(encoder_cache_params, encoder_memo_func)
        for param in param_names:
            setattr(encoder, param, copy.deepcopy(encoder_from_cache[param]))

        return encoder

    @staticmethod
    def check_dataset_type_available_in_mapping(dataset, class_name):
        if dataset.__class__.__name__ not in class_name.dataset_mapping.keys():
            raise ValueError(f"{class_name.__name__}: this encoder is not defined for dataset of type {dataset.__class__.__name__}. "
                             f"Valid dataset types for this encoder are: {', '.join(list(class_name.dataset_mapping.keys()))}")

    @staticmethod
    def encode_element_dataset_labels(dataset: ElementDataset, label_config: LabelConfiguration):

        labels = {name: [] for name in label_config.get_labels_by_name()}

        for sequence in dataset.get_data():
            for label_name in label_config.get_labels_by_name():
                label = sequence.get_attribute(label_name)
                labels[label_name].append(label)

        return labels

    @staticmethod
    def check_positive_class_labels(label_config: LabelConfiguration, location: str):
        '''
        Performs checks for Encoders that explicitly predict a positive class. These Encoders can only be trained for a
        single binary label at a time.
        '''

        labels = label_config.get_label_objects()
        assert len(labels) == 1, f"{location}: this encoding works only for single label."

        label = labels[0]

        assert isinstance(label, Label) and label.positive_class is not None and label.positive_class != "", \
            f"{location}: positive_class parameter was not set for label {label}. It has to be set to determine the " \
            f"receptor sequences associated with the positive class. " \
            f"To use this encoder, in the label definition in the specification of the instruction, define " \
            f"the positive class for the label. See documentation for this encoder for more details."

        assert len(label.values) == 2, f"{location}: only binary classification (2 classes) is possible when extracting " \
                                       f"relevant sequences for the label, but got these classes for label {label.name} instead: {label.values}."


    @staticmethod
    def get_example_weights_by_identifiers(dataset, example_identifiers):
        weights = dataset.get_example_weights()

        if weights is not None:
            weights_dict = dict(zip(dataset.get_example_ids(), weights))

            return [weights_dict[identifier] for identifier in example_identifiers]

