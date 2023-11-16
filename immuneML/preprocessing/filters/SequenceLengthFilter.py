from multiprocessing import Pool
from pathlib import Path

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.preprocessing.filters.Filter import Filter
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class SequenceLengthFilter(Filter):
    """
    Removes sequences with length out of the predefined range.

    Specification arguments:

    - sequence_type (:py:obj:`~immuneML.environment.SequenceType.SequenceType`): Whether the sequences should be filtered on the nucleotide or amino acid level. Valid options are defined by the SequenceType enum.

    - min_len (int): minimum length of the sequence (sequences shorter than min_len will be removed); to not use min_len, set it to -1

    - max_len (int): maximum length of the sequence (sequences longer than max_len will be removed); to not use max_len, set it to -1

    YAML specification:

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

    def __init__(self, min_len: int, max_len: int, sequence_type: SequenceType, name: str = None):
        super().__init__()
        self._min_len = min_len
        self._max_len = max_len
        self._sequence_type = sequence_type
        self._name = name

    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_keys_present(list(kwargs.keys()), ['min_len', 'max_len', 'sequence_type'],
                                               SequenceLengthFilter.__name__, SequenceLengthFilter.__name__)
        ParameterValidator.assert_all_type_and_value([kwargs['min_len'], kwargs['max_len']], int, SequenceLengthFilter.__name__, 'length')

        if kwargs['max_len'] >= 0:
            assert kwargs['min_len'] <= kwargs['max_len'], f"{SequenceLengthFilter.__name__}: min_len must be less or equal to max_len."
        assert kwargs['min_len'] >= 0 or kwargs['max_len'] >= 0, f"{SequenceLengthFilter.__name__}: at least one of min_len and max_len has to be set."
        ParameterValidator.assert_sequence_type(kwargs, SequenceLengthFilter.__name__)

        return cls(min_len=kwargs['min_len'], max_len=kwargs['max_len'], sequence_type=SequenceType[kwargs['sequence_type'].upper()],
                   name=kwargs['name'] if 'name' in kwargs else SequenceLengthFilter.__name__)

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path, number_of_processes: int = 1) -> RepertoireDataset:
        if not isinstance(dataset, RepertoireDataset):
            raise NotImplementedError

        new_reps_path = PathBuilder.build(result_path / 'repertoires')
        arguments = [(repertoire, new_reps_path) for repertoire in dataset.repertoires]

        with Pool(number_of_processes) as pool:
            repertoires = pool.starmap(self._process_repertoire, arguments)

        return RepertoireDataset.build_from_objects(repertoires=repertoires, path=result_path)

    def _process_repertoire(self, repertoire: Repertoire, result_path: Path) -> Repertoire:
        sequences = repertoire.get_sequence_aas() if self._sequence_type == SequenceType.AMINO_ACID else repertoire.get_attribute('sequences')

        keep_seq_func = self._get_keep_seq_func()
        indices_to_keep = [keep_seq_func(seq) for seq in sequences]

        return Repertoire.build_like(repertoire, indices_to_keep, result_path,
                                     filename_base=repertoire.metadata['subject_id'] + '_filtered' if 'subject_id' in repertoire.metadata else None)

    def _get_keep_seq_func(self):
        if self._max_len < 0:
            return lambda x: len(x) >= self._min_len
        elif self._min_len < 0:
            return lambda x: len(x) <= self._max_len
        else:
            return lambda x: self._max_len >= len(x) >= self._min_len
