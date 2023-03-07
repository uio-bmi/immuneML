import logging
from pathlib import Path

import numpy as np

from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.dsl.import_parsers.ImportParser import ImportParser
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.generative_models.GenerativeModel import GenerativeModel
from immuneML.util.ParameterValidator import ParameterValidator


class ExperimentalImport(GenerativeModel):
    """
    Allows to import existing experimental data and do annotations and simulations on top of them.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        generative_model:
            import_format: AIRR
            import_params:
                path: path/to/files/
                region_type: IMGT_CDR3 # what part of the sequence to import
                column_mapping: # column mapping AIRR: immuneML
                    junction: sequence
                    junction_aa: sequence_aa
                    locus: chain
            type: ExperimentalImport
    """
    def __init__(self, dataset: SequenceDataset):
        self._dataset = dataset
        self._counter = -1

    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_keys(kwargs.keys(), ['import_format', 'import_params'], ExperimentalImport.__name__,  'ExperimentalImport')

        if 'is_repertoire' in kwargs['import_params']:
            assert kwargs['import_params']['is_repertoire'] is False, \
                f"{ExperimentalImport.__name__}: repertoire datasets cannot be imported for the purpose of simulation. " \
                f"Only sequence datasets are supported."
        else:
            kwargs['import_params']['is_repertoire'] = False

        dataset = ImportParser.parse_dataset("experimental_dataset", {'format': kwargs['import_format'], 'params': kwargs['import_params']})
        return ExperimentalImport(dataset)

    def load_model(self):
        pass

    def generate_sequences(self, count: int, seed: int, path: Path, sequence_type: SequenceType, compute_p_gen: bool):
        if compute_p_gen:
            logging.warning(f"{ExperimentalImport.__name__}: generation probabilities cannot be computed for experimental data, skipping...")

        if self._counter < self._dataset.get_example_count():
            sequences = self._dataset.get_data(batch_size=count)
        raise NotImplementedError

    def compute_p_gens(self, sequences, sequence_type: SequenceType) -> np.ndarray:
        raise NotImplementedError

    def compute_p_gen(self, sequence: dict, sequence_type: SequenceType) -> float:
        raise NotImplementedError

    def can_compute_p_gens(self) -> bool:
        return False

    def can_generate_from_skewed_gene_models(self) -> bool:
        return False

    def generate_from_skewed_gene_models(self, v_genes: list, j_genes: list, seed: int, path: Path, sequence_type: SequenceType, batch_size: int,
                                         compute_p_gen: bool):
        raise NotImplementedError
