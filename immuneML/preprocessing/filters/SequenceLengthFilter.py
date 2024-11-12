from multiprocessing import Pool
from pathlib import Path

import dill
import numpy as np

from immuneML.data_model import bnp_util
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.preprocessing.filters.Filter import Filter
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class SequenceLengthFilter(Filter):
    """
    Removes sequences with length out of the predefined range.

    **Specification arguments:**

    - sequence_type (:py:obj:`~immuneML.environment.SequenceType.SequenceType`): Whether the sequences should be filtered on the nucleotide or amino acid level. Valid options are defined by the SequenceType enum.

    - min_len (int): minimum length of the sequence (sequences shorter than min_len will be removed); to not use min_len, set it to -1

    - max_len (int): maximum length of the sequence (sequences longer than max_len will be removed); to not use max_len, set it to -1

    - region_type (str): which part of the sequence to examine, by default, this is IMGT_CDR3

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        preprocessing_sequences:
            my_preprocessing:
                - my_filter:
                    SequenceLengthFilter:
                        sequence_type: AMINO_ACID
                        min_len: 3 # -> remove all sequences shorter than 3
                        max_len: -1 # -> no upper bound on the sequence length

        """

    def __init__(self, min_len: int, max_len: int, sequence_type: SequenceType, region_type: RegionType, name: str = None):
        super().__init__()
        self._min_len = min_len
        self._max_len = max_len
        self._sequence_type = sequence_type
        self._region_type = region_type
        self._name = name

    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_keys_present(list(kwargs.keys()), ['min_len', 'max_len', 'sequence_type', 'region_type'],
                                               SequenceLengthFilter.__name__, SequenceLengthFilter.__name__)
        ParameterValidator.assert_all_type_and_value([kwargs['min_len'], kwargs['max_len']], int, SequenceLengthFilter.__name__, 'length')

        if kwargs['max_len'] >= 0:
            assert kwargs['min_len'] <= kwargs['max_len'], f"{SequenceLengthFilter.__name__}: min_len must be less or equal to max_len."
        assert kwargs['min_len'] >= 0 or kwargs['max_len'] >= 0, f"{SequenceLengthFilter.__name__}: at least one of min_len and max_len has to be set."
        ParameterValidator.assert_sequence_type(kwargs, SequenceLengthFilter.__name__)
        ParameterValidator.assert_region_type(kwargs, SequenceLengthFilter.__name__)

        return cls(min_len=kwargs['min_len'], max_len=kwargs['max_len'], sequence_type=SequenceType[kwargs['sequence_type'].upper()],
                   name=kwargs['name'] if 'name' in kwargs else SequenceLengthFilter.__name__, region_type=RegionType[kwargs['region_type']])

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path, number_of_processes: int = 1) -> RepertoireDataset:
        if not isinstance(dataset, RepertoireDataset):
            raise NotImplementedError

        new_reps_path = PathBuilder.build(result_path / 'repertoires')
        arguments = [(dill.dumps(repertoire), new_reps_path) for repertoire in dataset.repertoires]

        with Pool(number_of_processes) as pool:
            repertoires = pool.starmap(self._process_repertoire, arguments)

        return RepertoireDataset.build_from_objects(repertoires=repertoires, path=result_path)

    def _process_repertoire(self, repertoire: Repertoire, result_path: Path) -> Repertoire:
        repertoire = dill.loads(repertoire) if isinstance(repertoire, bytes) else repertoire
        sequences = getattr(repertoire.data, bnp_util.get_sequence_field_name(self._region_type, self._sequence_type))

        below_max_len = sequences.lengths <= self._max_len if self._max_len >= 0 else np.ones(len(sequences), dtype=bool)
        above_min_len = sequences.lengths >= self._min_len if self._min_len >= 0 else np.ones(len(sequences), dtype=bool)
        indices_to_keep = np.logical_and(above_min_len, below_max_len)

        return Repertoire.build_like(repertoire, indices_to_keep, result_path,
                                     filename_base=repertoire.metadata['subject_id'] + '_filtered' if 'subject_id' in repertoire.metadata else None)

    def _get_keep_seq_func(self):
        if self._max_len < 0:
            return lambda x: len(x) >= self._min_len
        elif self._min_len < 0:
            return lambda x: len(x) <= self._max_len
        else:
            return lambda x: self._max_len >= len(x) >= self._min_len
