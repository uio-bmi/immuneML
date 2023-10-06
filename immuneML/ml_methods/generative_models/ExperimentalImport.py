import logging
from pathlib import Path

import numpy as np

from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.dsl.import_parsers.ImportParser import ImportParser
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.BackgroundSequences import BackgroundSequences
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.simulation.util.util import write_bnp_data
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class ExperimentalImport(GenerativeModel):
    """
    Allows to import existing experimental data and do annotations and simulations on top of them.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        generative_model:
            import_format: AIRR
            tmp_import_path: ./tmp/
            import_params:
                path: path/to/files/
                region_type: IMGT_CDR3 # what part of the sequence to import
                column_mapping: # column mapping AIRR: immuneML
                    junction: sequence
                    junction_aa: sequence_aa
                    locus: chain
            type: ExperimentalImport

    """

    def __init__(self, dataset: SequenceDataset, original_input_file: Path = None):
        self._dataset = dataset
        self._counter = 0
        self._original_input_file = original_input_file

    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_keys(kwargs.keys(), ['import_format', 'import_params', "tmp_import_path"], ExperimentalImport.__name__,  'ExperimentalImport')
        ParameterValidator.assert_type_and_value(kwargs['tmp_import_path'], str, cls.__name__, 'tmp_import_path')
        tmp_import_path = Path(kwargs['tmp_import_path'])
        assert not tmp_import_path.is_file(), \
            f"{cls.__name__}: parameter 'tmp_import_path' has to point to a directory where temporary files can be stored."

        PathBuilder.build(tmp_import_path, False)

        dataset = ImportParser.parse_dataset("experimental_dataset", {'format': kwargs['import_format'], 'params': kwargs['import_params']},
                                             tmp_import_path)
        print(f"Imported dataset with {dataset.get_example_count()} sequences.")
        return ExperimentalImport(dataset, kwargs['import_params']['path'])

    def generate_sequences(self, count: int, seed: int, path: Path, sequence_type: SequenceType, compute_p_gen: bool):
        if compute_p_gen:
            logging.warning(f"{ExperimentalImport.__name__}: generation probabilities cannot be computed for experimental data, skipping...")

        if self._counter < self._dataset.get_example_count():
            sequences = self._dataset.get_data_from_index_range(self._counter, self._counter + count - 1)
            self._counter += len(sequences)
            write_bnp_data(path, BackgroundSequences.build_from_receptor_sequences(sequences))
        else:
            raise RuntimeError(f"{ExperimentalImport.__name__}: all sequences provided to the generative model were already used in the simulation, "
                               f"no more new sequences can be imported. Try increasing the number of sequences in the provided files or reduce the "
                               f"number of sequences or repertoires to be generated.")

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

    def is_same(self, model) -> bool:
        return type(self) == type(model) and self._dataset.get_example_count() == model._dataset.get_example_count() \
               and self._original_input_file == model._original_input_file
