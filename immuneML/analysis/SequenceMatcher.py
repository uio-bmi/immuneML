from multiprocessing.pool import Pool

import numpy as np
from editdistance import eval as edit_distance

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.reference_encoding.SequenceMatchingSummaryType import SequenceMatchingSummaryType


class SequenceMatcher:
    """
    Matches the sequences across the given list of reference sequences (a list of ReceptorSequence objects) and returns the following information:
    {
        "repertoires":[{
            "sequences": [{
                "sequence": "AAA",
                "matching_sequences": ["AAA", "AAC"],
                "v_gene": "V12",
                "j_gene": "J3",
                "chain": "A"
            }], # list of sequences for the repertoire with matched sequences for each original sequence
            "repertoire": "fdjshfk321231", # repertoire identifier
            "repertoire_index": 2,  # the index of the repertoire in the dataset,
            "sequences_matched": 4,  # number of sequences from the repertoire which are a match for at least one reference sequence
            "percentage_of_sequences_matched": 0.75,  # percentage of sequences from the repertoire that have at least one match in the reference sequences
            "metadata": {"CD": True},  # dict with parameters that can be used for analysis on repertoire level and that serve as a starting point for label configurations
            "chains": ["A","B"] # list of chains in the repertoire
        }, ...]
    }
    """

    CORES = 4

    def match(self, dataset: RepertoireDataset, reference_sequences: list, max_distance: int, summary_type: SequenceMatchingSummaryType) -> dict:

        matched = {"repertoires": []}

        for index, repertoire in enumerate(dataset.get_data()):
            matched["repertoires"].append(self.match_repertoire(repertoire, index,
                                                                reference_sequences, max_distance, summary_type))

        return matched

    def matches_gene(self, gene1, gene2):
        if gene1 == gene2:
            return True
        else:
            return gene2.split("-", 1)[0] == gene1 or gene1.split("-", 1)[0] == gene2

    def matches_sequence(self, original_sequence: ReceptorSequence, reference_sequence: ReceptorSequence, max_distance):
        """
        :param original_sequence: ReceptorSequence
        :param reference_sequence: ReceptorSequence
        :param max_distance: max allowed Levenshtein distance between two sequences to be considered a match
        :return: True if chain, v_gene and j_gene are the same and sequences are within given Levenshtein distance
        """
        return reference_sequence.metadata.chain == original_sequence.metadata.chain \
            and self.matches_gene(reference_sequence.metadata.v_gene, original_sequence.metadata.v_gene) \
            and self.matches_gene(reference_sequence.metadata.j_gene, original_sequence.metadata.j_gene) \
            and edit_distance(original_sequence.get_sequence(), reference_sequence.get_sequence()) <= max_distance

    def match_repertoire(self, repertoire: Repertoire, index: int, reference_sequences: list, max_distance: int,
                         summary_type: SequenceMatchingSummaryType) -> dict:

        matched = {"sequences": [], "repertoire": repertoire.identifier, "repertoire_index": index}
        arguments = [(seq, reference_sequences, max_distance) for seq in repertoire.sequences]

        with Pool(SequenceMatcher.CORES) as pool:
            matched["sequences"] = pool.starmap(self.match_sequence, arguments)

        if summary_type == SequenceMatchingSummaryType.CLONAL_PERCENTAGE:
            total_count = np.sum([sequence.metadata.count for sequence in repertoire.sequences])
            matched["clonal_percentage"] = np.sum([sequence.metadata.count for index, sequence in enumerate(repertoire.sequences) if len(matched["sequences"][index]["matching_sequences"]) > 0]) / total_count
        else:
            matched["count"] = len([r for r in matched["sequences"] if len(r["matching_sequences"]) > 0])
            matched["percentage"] = matched["count"] / len(matched["sequences"])
        matched["metadata"] = repertoire.metadata
        matched["patient_id"] = repertoire.identifier
        matched["chains"] = list(set([sequence.metadata.chain for sequence in repertoire.sequences]))

        return matched

    def match_sequence(self, sequence: ReceptorSequence, reference_sequences: list, max_distance: int) -> dict:
        matching_sequences = [seq.get_sequence() for seq in reference_sequences
                              if self.matches_sequence(sequence, seq, max_distance)]

        return {
            "matching_sequences": matching_sequences,
            "sequence": sequence.get_sequence(),
            "v_gene": sequence.metadata.v_gene,
            "j_gene": sequence.metadata.j_gene,
            "chain": sequence.metadata.chain
        }
