from pathlib import Path

import pandas as pd

from immuneML.data_model import bnp_util
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.environment.Constants import Constants
from immuneML.preprocessing.Preprocessor import Preprocessor
from immuneML.util.PathBuilder import PathBuilder


class SubjectRepertoireCollector(Preprocessor):
    """
    Merges all the Repertoires in a RepertoireDataset that have the same 'subject_id' specified in the metadata. The result
    is a RepertoireDataset with one Repertoire per subject. This preprocessing cannot be used in combination with :ref:`TrainMLModel`
    instruction because it can change the number of examples. To combine the repertoires in this way, use this preprocessing
    with :ref:`DatasetExport` instruction.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        preprocessing_sequences:
            my_preprocessing:
                - my_filter: SubjectRepertoireCollector

    """

    def __init__(self, result_path: Path = None):
        super().__init__(result_path)

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path, number_of_processes=1):
        self.result_path = PathBuilder.build(result_path if result_path is not None else self.result_path)
        self.check_dataset_type(dataset, [RepertoireDataset], "SubjectRepertoireCollector")

        processed_dataset = self._merge_repertoires(dataset)

        return processed_dataset

    def _merge_repertoires(self, dataset: RepertoireDataset):
        rep_map = {}
        repertoires, indices_to_keep = [], []
        processed_dataset = dataset.clone()

        for index, repertoire in enumerate(processed_dataset.get_data()):

            if repertoire.metadata["subject_id"] in rep_map.keys():
                rep_map[repertoire.metadata['subject_id']].append(repertoire)
            else:
                rep_map[repertoire.metadata['subject_id']] = [repertoire]

        for key in rep_map.keys():
            repertoires.append(self._store_repertoire(rep_map[key]))

        processed_dataset.repertoires = repertoires
        processed_dataset.metadata_file = self._build_new_metadata(dataset, indices_to_keep)

        return processed_dataset

    def _build_new_metadata(self, dataset, indices_to_keep):
        if dataset.metadata_file:
            df = pd.read_csv(dataset.metadata_file, index_col=0, comment=Constants.COMMENT_SIGN).iloc[indices_to_keep,
                 :]
            path = Path(self.result_path / f"{dataset.metadata_file.stem}_collected_repertoires.csv")
            df.to_csv(path)
        else:
            path = None
        return path

    def _store_repertoire(self, repertoires: list):
        metadata = {k: v for d in [r.metadata for r in repertoires] for k, v in d.items()}
        new_repertoire = Repertoire.build_from_dc_object(self.result_path, metadata,
                                                         data=bnp_util.merge_dataclass_objects(
                                                             [r.data for r in repertoires]))
        return new_repertoire

    def keeps_example_count(self) -> bool:
        return False
