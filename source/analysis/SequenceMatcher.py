from editdistance import eval as edit_distance

from source.data_model.dataset.Dataset import Dataset
from source.data_model.repertoire.Repertoire import Repertoire


class SequenceMatcher:
    """
    Matches the sequences across the given list of reference sequences (a list of strings) and returns the following information:
    {
        "repertoires":[{
            "sequences": ["AA", "ACAGTF"], # original list of sequences for the repertoire
            "repertoire": "fdjshfk321231", # repertoire identifier
            "repertoire_index": 2,  # the index of the repertoire in the dataset,
            "sequences_matched": 4,  # number of sequences from the repertoire which are a match for at least one reference sequence
            "percentage_of_sequences_matched": 0.75,  # percentage of sequences from the repertoire that have at least one match in the reference sequences
            "metadata": {"CD": True}  # dict with parameters that can be used for analysis on repertoire level and that serve as a starting point for label configurations
        }, ...]
    }
    """

    def match(self, dataset: Dataset, reference_sequences: list, max_distance: int) -> dict:

        matched = {"repertoires": []}

        for index, repertoire in enumerate(dataset.get_data()):
            matched["repertoires"].append(self.match_repertoire(repertoire, index, reference_sequences, max_distance))

        return matched

    def match_repertoire(self, repertoire: Repertoire, index: int, reference_sequences: list, max_distance: int) -> dict:

        matched = {"sequences": [], "repertoire": repertoire.identifier, "repertoire_index": index}

        for sequence in repertoire.sequences:
            matching_sequences = [seq for seq in reference_sequences
                                  if edit_distance(sequence.get_sequence(), seq) <= max_distance]

            matched["sequences"].append({
                "matching_sequences": matching_sequences,
                "sequence": sequence.get_sequence()
            })

        matched["sequences_matched"] = len([r for r in matched["sequences"] if len(r["matching_sequences"]) > 0])
        matched["percentage_of_sequences_matched"] = matched["sequences_matched"] / len(matched["sequences"])
        matched["metadata"] = repertoire.metadata.sample.custom_params \
            if repertoire.metadata is not None and repertoire.metadata.sample is not None else None

        return matched
