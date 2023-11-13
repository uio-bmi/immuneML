from typing import List

import numpy as np
import pandas as pd

from immuneML.analysis.SequenceMatcher import SequenceMatcher
from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.receptor.BCReceptor import BCReceptor
from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.receptor.TCABReceptor import TCABReceptor
from immuneML.data_model.receptor.TCGDReceptor import TCGDReceptor
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.reference_encoding.MatchedReferenceUtil import MatchedReferenceUtil
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReadsType import ReadsType


class MatchedReceptorsEncoder(DatasetEncoder):
    """
    Encodes the dataset based on the matches between a dataset containing unpaired (single chain) data,
    and a paired reference receptor dataset.
    For each paired reference receptor, the frequency of either chain in the dataset is counted.

    This encoding can be used in combination with the :ref:`Matches` report.

    When sum_matches and normalize are set to True, this encoder behaves similarly as described in: Yao, Y. et al. ‘T cell receptor repertoire as a potential diagnostic marker for celiac disease’.
    Clinical Immunology Volume 222 (January 2021): 108621. `doi.org/10.1016/j.clim.2020.108621 <https://doi.org/10.1016/j.clim.2020.108621>`_
    with the only exception being that this encoder uses paired receptors, while the original publication used single sequences (see also: :ref:`MatchedSequences` encoder).

    Specification arguments:

    - reference (dict): A dictionary describing the reference dataset file. Import should be specified the same way as
      regular dataset import. It is only allowed to import a receptor dataset here (i.e., is_repertoire is False and
      paired is True by default, and these are not allowed to be changed).

    - max_edit_distances (dict): A dictionary specifying the maximum edit distance between a target sequence (from the
      repertoire) and the reference sequence. A maximum distance can be specified per chain, for example to allow for
      less strict matching of TCR alpha and BCR light chains. When only an integer is specified, this distance is
      applied to all possible chains.

    - reads (:py:mod:`~immuneML.util.ReadsType`): Reads type signify whether the counts of the sequences in the
      repertoire will be taken into account. If :py:mod:`~immuneML.util.ReadsType.UNIQUE`, only unique sequences
      (clonotypes) are counted, and if :py:mod:`~immuneML.util.ReadsType.ALL`, the sequence 'count' value is summed when
      determining the number of matches. The default value for reads is all.

    - sum_matches (bool): When sum_matches is False, the resulting encoded data matrix contains multiple columns with
      the number of matches per reference receptor chain. When sum_matches is true, the columns representing each of the
      two chains are summed together, meaning that there are only two aggregated sums of matches (one per chain) per
      repertoire in the encoded data. To use this encoder in combination with the :ref:`Matches` report, sum_matches
      must be set to False. When sum_matches is set to True, this encoder behaves similarly to the encoder described by
      Yao, Y. et al. By default, sum_matches is False.

    - normalize (bool): If True, the chain matches are divided by the total number of unique receptors in the repertoire
      (when reads = unique) or the total number of reads in the repertoire (when reads = all).


    YAML Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_mr_encoding:
            MatchedReceptors:
                reference:
                    format: VDJDB
                    params:
                        path: path/to/file.txt
                max_edit_distances:
                    alpha: 1
                    beta: 0
    """

    dataset_mapping = {
        "RepertoireDataset": "MatchedReceptorsRepertoireEncoder"
    }

    def __init__(self, reference: List[Receptor], max_edit_distances: dict, reads: ReadsType, sum_matches: bool, normalize: bool, name: str = None):
        self.reference_receptors = reference
        self.max_edit_distances = max_edit_distances
        self.reads = reads
        self.sum_matches = sum_matches
        self.normalize = normalize
        self.feature_count = 2 if self.sum_matches else len(self.reference_receptors) * 2
        self.name = name

    @staticmethod
    def _prepare_parameters(reference: dict, max_edit_distances: dict, reads: str, sum_matches: bool, normalize: bool, name: str = None):
        location = "MatchedReceptorsEncoder"

        ParameterValidator.assert_type_and_value(sum_matches, bool, location, "sum_matches")
        ParameterValidator.assert_type_and_value(normalize, bool, location, "normalize")
        ParameterValidator.assert_in_valid_list(reads.upper(), [item.name for item in ReadsType], location, "reads")

        legal_chains = [chain for receptor in (TCABReceptor(), TCGDReceptor(), BCReceptor()) for chain in receptor.get_chains()]

        if type(max_edit_distances) is int:
            max_edit_distances = {chain: max_edit_distances for chain in legal_chains}
        elif type(max_edit_distances) is dict:
            ParameterValidator.assert_keys(max_edit_distances.keys(), legal_chains, location, "max_edit_distances", exclusive=False)
        else:
            ParameterValidator.assert_type_and_value(max_edit_distances, dict, location, 'max_edit_distances')

        reference_receptors = MatchedReferenceUtil.prepare_reference(reference, location=location, paired=True)

        return {
            "reference": reference_receptors,
            "max_edit_distances": max_edit_distances,
            "reads": ReadsType[reads.upper()],
            "sum_matches": sum_matches,
            "normalize": normalize,
            "name": name
        }

    @staticmethod
    def build_object(dataset=None, **params):
        if isinstance(dataset, RepertoireDataset):
            prepared_parameters = MatchedReceptorsEncoder._prepare_parameters(**params)
            return MatchedReceptorsEncoder(**prepared_parameters)
        else:
            raise ValueError("MatchedSequencesEncoder is not defined for dataset types which are not RepertoireDataset.")

    def encode(self, dataset, params: EncoderParams):
        cache_key = CacheHandler.generate_cache_key(self._prepare_caching_params(dataset, params))
        encoded_dataset = CacheHandler.memo(cache_key,
                                            lambda: self._encode_new_dataset(dataset, params))

        return encoded_dataset

    def _prepare_caching_params(self, dataset, params: EncoderParams):

        chains = [(receptor.get_chain(receptor.get_chains()[0]), receptor.get_chain(receptor.get_chains()[1]))
                  for receptor in self.reference_receptors]

        encoding_params_desc = {"max_edit_distance": sorted(self.max_edit_distances.items()),
                                "reference_receptors": sorted([chain_a.get_sequence() + chain_a.metadata.v_gene +
                                                               chain_a.metadata.j_gene + "|" + chain_b.get_sequence() +
                                                               chain_b.metadata.v_gene + chain_b.metadata.j_gene
                                                               for chain_a, chain_b in chains]),
                                "reads": self.reads.name,
                                "sum_matches": self.sum_matches,
                                "normalize": self.normalize}

        return (("dataset_identifiers", tuple(dataset.get_example_ids())),
                ("dataset_metadata", dataset.metadata_file),
                ("dataset_type", dataset.__class__.__name__),
                ("labels", tuple(params.label_config.get_labels_by_name())),
                ("encoding", MatchedReceptorsEncoder.__name__),
                ("learn_model", params.learn_model),
                ("encoding_params", encoding_params_desc), )

    def _encode_new_dataset(self, dataset, params: EncoderParams):
        pass

        encoded_dataset = RepertoireDataset(repertoires=dataset.repertoires, labels=dataset.labels,
                                            metadata_file=dataset.metadata_file)

        feature_annotations = None if self.sum_matches else self._get_feature_info()

        if self.sum_matches:
            chains = self.reference_receptors[0].get_chains()
            feature_names = [f"sum_of_{self.reads.value}_reads_{chains[0]}", f"sum_of_{self.reads.value}_reads_{chains[1]}"]
        else:
            feature_names = [f"{row['receptor_id']}.{row['chain']}" for index, row in feature_annotations.iterrows()]

        encoded_repertoires, labels, example_ids = self._encode_repertoires(dataset, params)
        encoded_repertoires = self._normalize(dataset, encoded_repertoires) if self.normalize else encoded_repertoires

        encoded_dataset.add_encoded_data(EncodedData(
            # examples contains a np.ndarray with counts
            examples=encoded_repertoires,
            # example_ids contains a list of repertoire identifiers
            example_ids=example_ids,
            # feature_names contains a list of reference receptor identifiers
            feature_names=feature_names,
            # feature_annotations contains a PD dataframe with sequence and VDJ gene usage per reference receptor
            feature_annotations=feature_annotations,
            labels=labels,
            encoding=MatchedReceptorsEncoder.__name__
        ))

        return encoded_dataset

    def _normalize(self, dataset, encoded_repertoires):
        if self.reads == ReadsType.UNIQUE:
            repertoire_totals = np.asarray([[repertoire.get_element_count() for repertoire in dataset.get_data()]]).T
        else:
            repertoire_totals = np.asarray([[sum(repertoire.get_counts()) for repertoire in dataset.get_data()]]).T

        return encoded_repertoires / repertoire_totals


    def _get_feature_info(self):
        """
        returns a pandas dataframe containing:
         - receptor id
         - receptor chain
         - amino acid sequence
         - v gene
         - j gene
        """

        features = [[] for i in range(0, self.feature_count)]

        for i, receptor in enumerate(self.reference_receptors):
            id = receptor.identifier
            chain_names = receptor.get_chains()
            first_chain = receptor.get_chain(chain_names[0])
            second_chain = receptor.get_chain(chain_names[1])

            clonotype_id = receptor.metadata["clonotype_id"] if "clonotype_id" in receptor.metadata else None

            if first_chain.metadata.custom_params is not None:
                first_dual_chain_id = first_chain.metadata.custom_params["dual_chain_id"] if "dual_chain_id" in first_chain.metadata.custom_params else None

            if second_chain.metadata.custom_params is not None:
                second_dual_chain_id = second_chain.metadata.custom_params["dual_chain_id"] if "dual_chain_id" in second_chain.metadata.custom_params else None

            features[i * 2] = [id, clonotype_id, chain_names[0],
                               first_dual_chain_id,
                               first_chain.sequence_aa,
                               first_chain.metadata.v_gene,
                               first_chain.metadata.j_gene]
            features[i * 2 + 1] = [id, clonotype_id, chain_names[1],
                                   second_dual_chain_id,
                                   second_chain.sequence_aa,
                                   second_chain.metadata.v_gene,
                                   second_chain.metadata.j_gene]

        features = pd.DataFrame(features,
                                columns=["receptor_id", "clonotype_id", "chain", "dual_chain_id", "sequence", "v_gene", "j_gene"])

        features.dropna(axis="columns", how="all", inplace=True)

        return features

    def _encode_repertoires(self, dataset: RepertoireDataset, params: EncoderParams):
        # Rows = repertoires, Columns = reference chains (two per sequence receptor)
        encoded_repertories = np.zeros((dataset.get_example_count(),
                                        self.feature_count),
                                       dtype=int)
        labels = {label: [] for label in params.label_config.get_labels_by_name()} if params.encode_labels else None

        for i, repertoire in enumerate(dataset.get_data()):
            encoded_repertories[i] = self._compute_matches_to_reference(repertoire)

            if labels is not None:
                for label_name in params.label_config.get_labels_by_name():
                    labels[label_name].append(repertoire.metadata[label_name])

        return encoded_repertories, labels, dataset.get_repertoire_ids()

    def _get_repertoire_matches_to_reference(self, repertoire):
         return CacheHandler.memo_by_params(
             (("repertoire_identifier", repertoire.identifier),
              ("encoding", MatchedReceptorsEncoder.__name__),
              ("readstype", self.reads.name),
              ("sum_matches", self.sum_matches),
              ("max_edit_distances", tuple(self.max_edit_distances.items())),
              ("reference_receptors", tuple([self.get_receptor_params(receptor) for receptor in self.reference_receptors]))),
            lambda: self._compute_matches_to_reference(repertoire))

    def get_receptor_params(self, receptor):
        params = []

        for chain in receptor.get_chains():
            receptor_sequence = receptor.get_chain(chain)
            params.append((chain, receptor_sequence.get_sequence(), receptor_sequence.get_attribute("v_gene"), receptor_sequence.get_attribute("j_gene")))

        return tuple(params)

    def _compute_matches_to_reference(self, repertoire: Repertoire):
        matcher = SequenceMatcher()
        matches = np.zeros(self.feature_count, dtype=int)
        rep_seqs = repertoire.sequences

        for i, ref_receptor in enumerate(self.reference_receptors):
            chain_names = ref_receptor.get_chains()
            first_chain = ref_receptor.get_chain(chain_names[0])
            second_chain = ref_receptor.get_chain(chain_names[1])

            for rep_seq in rep_seqs:
                matches_idx = 0 if self.sum_matches else i * 2
                match_count = 1 if self.reads == ReadsType.UNIQUE else rep_seq.metadata.duplicate_count

                # Match with first chain: add to even columns in matches.
                # Match with second chain: add to odd columns
                if matcher.matches_sequence(first_chain, rep_seq, max_distance=self.max_edit_distances[chain_names[0]]):
                    matches[matches_idx] += match_count
                if matcher.matches_sequence(second_chain, rep_seq, max_distance=self.max_edit_distances[chain_names[1]]):
                    matches[matches_idx + 1] += match_count

        return matches
