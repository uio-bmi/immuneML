from multiprocessing.pool import Pool
from typing import List

import dill
import numpy as np
import pandas as pd

from immuneML.analysis.SequenceMatcher import SequenceMatcher
from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.reference_encoding.MatchedReferenceUtil import MatchedReferenceUtil
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReadsType import ReadsType


class MatchedSequencesEncoder(DatasetEncoder):
    """
    Encodes the dataset based on the matches between a RepertoireDataset and a reference sequence dataset.

    This encoding can be used in combination with the :ref:`Matches` report.

    When sum_matches and normalize are set to True, this encoder behaves as described in: Yao, Y. et al. ‘T cell receptor repertoire as a potential diagnostic marker for celiac disease’.
    Clinical Immunology Volume 222 (January 2021): 108621. `doi.org/10.1016/j.clim.2020.108621 <https://doi.org/10.1016/j.clim.2020.108621>`_


    **Dataset type:**

    - RepertoireDatasets


    **Specification arguments:**

    - reference (dict): A dictionary describing the reference dataset file. Import should be specified the same way as
      regular dataset import. It is only allowed to import a sequence dataset here (i.e., is_repertoire and paired are
      False by default, and are not allowed to be set to True).

    - max_edit_distance (int): The maximum edit distance between a target sequence (from the repertoire) and the
      reference sequence.

    - reads (:py:mod:`~immuneML.util.ReadsType`): Reads type signify whether the counts of the sequences in the
      repertoire will be taken into account. If :py:mod:`~immuneML.util.ReadsType.UNIQUE`, only unique sequences
      (clonotypes) are counted, and if :py:mod:`~immuneML.util.ReadsType.ALL`, the sequence 'count' value is summed when
      determining the number of matches. The default value for reads is all.

    - sum_matches (bool): When sum_matches is False, the resulting encoded data matrix contains multiple columns with
      the number of matches per reference sequence. When sum_matches is true, all columns are summed together, meaning
      that there is only one aggregated sum of matches per repertoire in the encoded data.
      To use this encoder in combination with the :ref:`Matches` report, sum_matches must be set to False. When
      sum_matches is set to True, this encoder behaves as described by Yao, Y. et al. By default, sum_matches is False.

    - normalize (bool): If True, the sequence matches are divided by the total number of unique sequences in the
      repertoire (when reads = unique) or the total number of reads in the repertoire (when reads = all).

    - output_count_as_feature: if True, the encoded repertoire is represented by the matches, and by the total number
      of sequences (or reads) in the repertoire, as defined by reads parameter above; by default this is False


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                my_ms_encoding:
                    MatchedSequences:
                        reference:
                            format: VDJDB
                            params:
                                path: path/to/file.txt
                        max_edit_distance: 1
    """

    def __init__(self, max_edit_distance: int, reference: List[ReceptorSequence], reads: ReadsType, sum_matches: bool,
                 normalize: bool, output_count_as_feature: bool = False, name: str = None):
        super().__init__(name=name)
        self.max_edit_distance = max_edit_distance
        self.reference_sequences = reference
        self.reads = reads
        self.sum_matches = sum_matches
        self.normalize = normalize
        self.output_count_as_feature = output_count_as_feature
        self.feature_count = (1 if self.sum_matches else len(self.reference_sequences)) + int(output_count_as_feature)

    @staticmethod
    def _prepare_parameters(max_edit_distance: int, reference: dict, reads: str, sum_matches: bool, normalize: bool,
                            output_count_as_feature: bool = False, name: str = None):
        location = "MatchedSequencesEncoder"

        ParameterValidator.assert_type_and_value(max_edit_distance, int, location, "max_edit_distance", min_inclusive=0)
        ParameterValidator.assert_type_and_value(sum_matches, bool, location, "sum_matches")
        ParameterValidator.assert_type_and_value(output_count_as_feature, bool, location, "normalize")
        ParameterValidator.assert_type_and_value(normalize, bool, location, "output_count_as_feature")
        ParameterValidator.assert_in_valid_list(reads.upper(), [item.name for item in ReadsType], location, "reads")

        if output_count_as_feature and normalize:
            raise RuntimeError(f"{MatchedSequencesEncoder.__name__}: normalize and output_count_as_feature cannot \n"
                               f"be both set to True at the same time. The sequence count (or reads count) can either \n"
                               f"be used for normalization or included as a separate output.")

        reference_sequences = MatchedReferenceUtil.prepare_reference(reference_params=reference, location=location, paired=False)

        return {
            "max_edit_distance": max_edit_distance,
            "reference": reference_sequences,
            "reads": ReadsType[reads.upper()],
            "sum_matches": sum_matches,
            "normalize": normalize,
            "output_count_as_feature": output_count_as_feature,
            "name": name
        }

    @staticmethod
    def build_object(dataset=None, **params):
        if isinstance(dataset, RepertoireDataset):
            prepared_parameters = MatchedSequencesEncoder._prepare_parameters(**params)
            return MatchedSequencesEncoder(**prepared_parameters)
        else:
            raise ValueError("MatchedSequencesEncoder is not defined for dataset types which are not RepertoireDataset.")

    def encode(self, dataset, params: EncoderParams):

        cache_key = CacheHandler.generate_cache_key(self._prepare_caching_params(dataset, params))
        encoded_dataset = CacheHandler.memo(cache_key,
                                            lambda: self._encode_new_dataset(dataset, params))

        return encoded_dataset

    def _prepare_caching_params(self, dataset, params: EncoderParams):

        encoding_params_desc = {"max_edit_distance": self.max_edit_distance,
                                "reference_sequences": sorted([seq.get_sequence() + seq.v_call + seq.j_call
                                                               for seq in self.reference_sequences]),
                                "reads": self.reads.name,
                                "sum_matches": self.sum_matches,
                                "normalize": self.normalize,
                                "output_count_as_feature": self.output_count_as_feature}

        return (("dataset_identifiers", tuple(dataset.get_example_ids())),
                ("dataset_metadata", dataset.metadata_file),
                ("dataset_type", dataset.__class__.__name__),
                ("labels", tuple(params.label_config.get_labels_by_name())),
                ("encoding", MatchedSequencesEncoder.__name__),
                ("learn_model", params.learn_model),
                ("encoding_params", encoding_params_desc),)

    def _encode_new_dataset(self, dataset, params: EncoderParams):
        encoded_repertoires, labels = self._encode_repertoires(dataset, params)

        encoded_repertoires = self._normalize(dataset, encoded_repertoires) if self.normalize else encoded_repertoires

        feature_annotations = None if self.sum_matches else self._get_feature_info()
        if self.sum_matches:
            feature_names = [f"sum_of_{self.reads.value}_reads"]
            if self.output_count_as_feature:
                feature_names += ['sequence_count_in_repertoire']
        else:
            feature_names = list(feature_annotations["sequence_desc"])

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(
            examples=encoded_repertoires,
            labels=labels,
            feature_names=feature_names,
            feature_annotations=feature_annotations,
            example_ids=[repertoire.identifier for repertoire in dataset.get_data()],
            encoding=MatchedSequencesEncoder.__name__,
            info={'sequence_type': params.sequence_type,
                  'region_type': params.region_type}
        )

        return encoded_dataset

    def _normalize(self, dataset, encoded_repertoires):
        if self.reads == ReadsType.UNIQUE:
            repertoire_totals = np.asarray([[repertoire.get_element_count() for repertoire in dataset.get_data()]]).T
        else:
            repertoire_totals = np.asarray([[sum(repertoire.data.duplicate_count) for repertoire in dataset.get_data()]]).T

        return encoded_repertoires / repertoire_totals

    def _get_feature_info(self):
        """
        returns a pandas dataframe containing:
         - sequence id
         - chain
         - amino acid sequence
         - v call
         - j call
        """

        features = [[] for i in range(0, len(self.reference_sequences))]

        for i, sequence in enumerate(self.reference_sequences):
            features[i] = [sequence.sequence_id,
                           sequence.locus,
                           sequence.sequence_aa,
                           sequence.v_call,
                           sequence.j_call,
                           self._get_sequence_desc(sequence)]

        if self.output_count_as_feature:
            features += [['', '', '', '', '', 'sequence_count_in_repertoire']]

        features = pd.DataFrame(features, columns=["sequence_id", "locus", "sequence", "v_call", "j_call", "sequence_desc"])
        if features['sequence_desc'].unique().shape[0] < features.shape[0]:
            features.loc[:, 'sequence_desc'] = [row['sequence_desc'] + "_" + row['sequence_id'] for ind, row in features.iterrows()]

        return features

    def _get_sequence_desc(self, sequence: ReceptorSequence) -> str:
        desc = ""
        if sequence.v_call not in [None, ""]:
            desc += f"{sequence.v_call}_"

        desc += sequence.sequence_aa if sequence.sequence_aa != "" else sequence.sequence

        if sequence.j_call not in ["", None]:
            desc += f"_{sequence.j_call}"

        return desc

    def _encode_repertoires(self, dataset: RepertoireDataset, params):
        labels = {label: [] for label in params.label_config.get_labels_by_name()} if params.encode_labels else None

        with Pool(params.pool_size) as pool:
            encoded_repertories = np.array(pool.map(self._get_repertoire_matches_to_reference,
                                                    [dill.dumps(rep) for rep in dataset.repertoires]))

        for repertoire in dataset.repertoires:
            for label_name in params.label_config.get_labels_by_name():
                labels[label_name].append(repertoire.metadata[label_name])

        return encoded_repertories, labels

    def _get_repertoire_matches_to_reference(self, repertoire):

        if isinstance(repertoire, bytes):
            repertoire = dill.loads(repertoire)

        return CacheHandler.memo_by_params(
            (("repertoire_identifier", repertoire.identifier),
             ("encoding", MatchedSequencesEncoder.__name__),
             ("readstype", self.reads.name),
             ("sum_matches", self.sum_matches),
             ("max_edit_distance", self.max_edit_distance),
             ("reference_sequences", tuple(
                 [(seq.locus, seq.sequence_aa, seq.v_call, seq.j_call) for seq in self.reference_sequences]))),
            lambda: self._compute_matches_to_reference(repertoire))

    def _compute_matches_to_reference(self, repertoire: Repertoire):
        matcher = SequenceMatcher()
        matches = np.zeros(self.feature_count, dtype=int)
        rep_seqs = repertoire.sequences()

        for i, reference_seq in enumerate(self.reference_sequences):

            for repertoire_seq in rep_seqs:
                if matcher.matches_sequence(reference_seq, repertoire_seq, max_distance=self.max_edit_distance):
                    matches_idx = 0 if self.sum_matches else i
                    match_count = 1 if self.reads == ReadsType.UNIQUE else repertoire_seq.duplicate_count
                    matches[matches_idx] += match_count

        if self.output_count_as_feature:
            duplicate_counts = repertoire.data.duplicate_count
            matches[-1] = sum(duplicate_counts) if self.reads == ReadsType.ALL else len(duplicate_counts)

        return matches
