import uuid
from pathlib import Path
from uuid import uuid4

from immuneML.data_model.SequenceSet import SequenceSet
from immuneML.data_model.bnp_util import make_buffer_type_from_dataset_file, write_yaml, read_yaml
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.receptor.ElementGenerator import ElementGenerator


class ElementDataset(Dataset):
    """
    This is the base class for ReceptorDataset and SequenceDataset which implements all the functionality for both classes. The only difference between
    these two classes is whether paired or single chain data is stored.
    """

    @classmethod
    def build(cls, dataset_file: Path, types: dict = None, filenames: list = None, **kwargs):
        if not Path(dataset_file).exists():
            metadata = {
                'type_dict': {key: SequenceSet.TYPE_TO_STR[val] for key, val in types.items()},
                'dataset_class': cls.__name__, 'element_class_name': kwargs['element_class_name'],
                'filenames': [str(file) for file in filenames]
            }
            write_yaml(dataset_file, metadata)
        return cls(**{**kwargs, 'dataset_file': dataset_file, 'filenames': filenames})

    def __init__(self, labels: dict = None, encoded_data: EncodedData = None, filenames: list = None,
                 identifier: str = None, dataset_file: Path = None,
                 file_size: int = 100000, name: str = None, element_class_name: str = None,
                 element_ids: list = None, example_weights: list = None,
                 buffer_type=None):
        super().__init__(encoded_data, name, identifier if identifier is not None else uuid4().hex, labels, example_weights)
        self.filenames = filenames if filenames is not None else []
        self.filenames = [Path(filename) for filename in self.filenames]
        if buffer_type is None:
            buffer_type = make_buffer_type_from_dataset_file(Path(dataset_file))
        self.element_generator = ElementGenerator(self.filenames, file_size, element_class_name, buffer_type)
        self.file_size = file_size
        self.element_ids = element_ids
        self.element_class_name = element_class_name
        self.dataset_file = Path(dataset_file)

    def get_data(self, batch_size: int = 10000, return_objects: bool = True):
        self.element_generator.file_list = self.filenames
        return self.element_generator.build_element_generator(return_objects=return_objects)

    def get_batch(self, batch_size: int = 10000):
        self.element_generator.file_list = self.filenames
        return self.element_generator.build_batch_generator()

    def get_filenames(self):
        return self.filenames

    def set_filenames(self, filenames):
        self.filenames = filenames

    def get_example_count(self):
        return len(self.get_example_ids())

    def get_example_ids(self):
        if self.element_ids is None or (isinstance(self.element_ids, list) and len(self.element_ids) == 0):
            self.element_ids = []
            for element in self.get_data():
                self.element_ids.append(str(element.get_id()))
        return self.element_ids

    def get_attribute(self, attribute: str, as_list: bool = False):
        res = self.element_generator.get_attribute(attribute)
        if as_list:
            return res.tolist()
        else:
            return res

    def get_attributes(self, attributes: list, as_list: bool = False) -> dict:
        res = self.element_generator.get_attributes(attributes)
        if as_list:
            return {attr: val.tolist() for attr, val in res.items()}
        else:
            return res

    def make_subset(self, example_indices, path, dataset_type: str):
        """
        Creates a new dataset object with only those examples (receptors or receptor sequences) available which were given by index in example_indices argument.

        Args:
            example_indices (list): a list of indices of examples (receptors or receptor sequences) to use in the new dataset
            path (Path): a path where to store the newly created dataset
            dataset_type (str): a type of the dataset used as a part of the name of the resulting dataset; the values are defined as constants in :py:obj:`~immuneML.data_model.dataset.Dataset.Dataset`

        Returns:

            a new dataset object (ReceptorDataset or SequenceDataset, as the original dataset) which includes only the examples specified under example_indices

        """
        new_dataset_id = uuid.uuid4().hex

        batch_filenames = self.element_generator.make_subset(example_indices, path, dataset_type, new_dataset_id,
                                                             paired=self.element_class_name != 'ReceptorSequence')
        dataset_name = f"{self.name}_split_{dataset_type.lower()}"

        types = read_yaml(self.dataset_file)['type_dict']

        new_dataset = self.__class__.build(labels=self.labels, file_size=self.file_size, filenames=batch_filenames,
                                           element_class_name=self.element_generator.element_class_name,
                                           dataset_file=path / f"{dataset_name}.yaml", types=types,
                                           identifier=new_dataset_id, name=dataset_name)

        # todo check if this is necessary
        original_example_weights = self.get_example_weights()
        if original_example_weights is not None:
            new_dataset.set_example_weights([original_example_weights[i] for i in example_indices])

        return new_dataset

    def get_label_names(self):
        """Returns the list of metadata fields which can be used as labels"""
        return [label for label in list(self.labels.keys()) if
                label not in ['region_type', 'receptor_chains', 'organism']] if self.labels else []

    def clone(self, keep_identifier: bool = False):
        raise NotImplementedError

    def get_data_from_index_range(self, start_index: int, end_index: int):
        return self.element_generator.get_data_from_index_range(start_index, end_index)
