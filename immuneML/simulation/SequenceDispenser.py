import logging
import random
from collections import defaultdict

import pandas as pd

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.generative_models.OLGA import OLGA
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation


class SequenceDispenser:
    def __init__(self, dataset: RepertoireDataset, occurrence_limit_pgen_range: dict = None,
                 mutation_hamming_distance: int = 1):

        self.seeds = []

        # dict with (key, value) = (mutation seq., list of repertoire ids.)
        self._mutation_implanted_repertoires = {}
        self._mutation_occurrence_limit = {}

        model_name = SequenceDispenser.get_default_model_name(dataset)

        self.olga = OLGA.build_object(model_path=None, default_model_name=model_name, chain=None,
                                      use_only_productive=False)
        self.olga.load_model()

        self.occurrence_limit_pgen_range = occurrence_limit_pgen_range

        self.mutation_hamming_distance = mutation_hamming_distance
        assert self.mutation_hamming_distance > 0, f"Mutation hamming distance must be greater than 0: {self.mutation_hamming_distance=}"

        if not self.occurrence_limit_pgen_range:
            self.total_sequence_occurrence = self._total_sequence_occurrence(dataset)

        self.amino_alphabet = EnvironmentSettings.get_sequence_alphabet(sequence_type=SequenceType.AMINO_ACID)

    def add_seed_sequence(self, seed_motif: Motif):
        """
        Add a full sequence from a Motif-object to the Sequence Dispenser

        Args:
            seed_motif (Motif): Motif-object with full sequence
        """
        if seed_motif in self.seeds:
            return

        assert seed_motif.v_call, f"Motif must have v_call: {seed_motif}"
        assert seed_motif.j_call, f"Motif must have j_call: {seed_motif}"

        assert self.mutation_hamming_distance < len(
            seed_motif.seed), f"Seed sequence length must be greater than mutation hamming distance: " \
                              f"seed sequence length={len(seed_motif.seed)}, " \
                              f"mutation hamming distance={self.mutation_hamming_distance}"

        seed_pgen = self.olga.compute_p_gens(
            pd.DataFrame(data={
                "sequence_aas": [seed_motif.seed],
                "v_genes": [seed_motif.v_call],
                "j_genes": [seed_motif.j_call]
            })
            , SequenceType.AMINO_ACID)[0]

        # check that seed pgen is > 0
        if seed_pgen == 0:
            logging.warning(
                f"Seed sequence \"{seed_motif.seed}\" has generation probability 0 and is incompatible "
                f"as a seed sequence for the current signal implanting strategy.")

        if seed_motif.mutation_position_possibilities is not None:
            mutation_position_possibilities = {key: float(value) for key, value in
                                               seed_motif.mutation_position_possibilities.items()}

            assert all(isinstance(key, int) for key in mutation_position_possibilities.keys()), \
                f"All keys in mutation_position_possibilities must have type int. {mutation_position_possibilities=}"
            assert all(isinstance(val, float) for val in mutation_position_possibilities.values()), \
                f"All values in mutation_position_possibilities must have type float. {mutation_position_possibilities=}"
            assert 0.99 <= round(sum(mutation_position_possibilities.values()), 5) <= 1, \
                f"For each possible mutation position a probability between 0 and 1 has to be assigned so that the " \
                f"probabilities for all distance possibilities sum to 1. sum({mutation_position_possibilities.values()}) = {sum(mutation_position_possibilities.values())}"

            assert all(0 <= i < len(seed_motif.seed) for i in mutation_position_possibilities.keys()), \
                f"All positions in {mutation_position_possibilities=} must be inside the seed sequence (i.e. in in " \
                f"range 0 to {len(seed_motif.seed) - 1}"
        else:
            raise ValueError("SequenceDispenser: Motifs must have mutation_position_possibilities")

        self.seeds.append(seed_motif)

    def generate_mutation(self, repertoire_id) -> Motif:
        """
        Generate a mutated sequence using added sequences

        Args:
            repertoire_id: id

        Returns:
            Motif: Motif-object with a randomly selected legal amino acid sequence
        """

        assert self.seeds, "SequenceDispenser: No signals to mutate"

        seed_motif = random.choice(self.seeds)

        seed_seq = seed_motif.seed
        v_call = seed_motif.v_call
        j_call = seed_motif.j_call

        mutation_position_probabilities = seed_motif.mutation_position_possibilities

        attempt_counter = 1

        while True:
            mutation = self._mutate_sequence(seed_seq, mutation_position_probabilities)

            df = pd.DataFrame(data={
                "sequence_aas": [mutation],
                "v_genes": [v_call],
                "j_genes": [j_call]
            })

            pgen = self.olga.compute_p_gens(df, SequenceType.AMINO_ACID)[0]

            if self._legal_mutation(mutation, pgen, repertoire_id):
                break

            attempt_counter += 1
            if attempt_counter > 200:
                # seed is removed if no legal mutation is found
                self.seeds.remove(seed_motif)
                logging.warning(
                    f"SequenceDispenser: No legal mutation found for {seed_seq} (v-gene={seed_motif.v_call}, "
                    f"j-gene={seed_motif.j_call}). Removing seed from selection list.")

                if not self.seeds:
                    raise Exception(
                        "SequenceDispenser: No legal mutations found for given signal seed sequences. Terminating simulation.")

                return self.generate_mutation(repertoire_id)

        print(f"      Mutation: {mutation}")
        return Motif(identifier=f"mutation_{mutation}", instantiation=GappedKmerInstantiation(), seed=mutation,
                     v_call=v_call,
                     j_call=j_call)

    def _mutate_sequence(self, seq: str, mutation_position_probabilities: dict):

        mutated_seq = seq

        for _ in range(self.mutation_hamming_distance):
            mutated_seq = self._mutate_one_position_in_sequence(mutated_seq, mutation_position_probabilities)

        return mutated_seq

    def _mutate_one_position_in_sequence(self, seq: str, mutation_position_probabilities: dict):
        mutation_position = random.choices(list(mutation_position_probabilities.keys()),
                                           weights=list(mutation_position_probabilities.values()), k=1)[0]
        amino_acid = random.choice(self.amino_alphabet)

        mutation = seq[:mutation_position] + amino_acid + seq[mutation_position + 1:]

        return mutation

    def get_seq_occurrences(self, seq: str):
        if seq not in self._mutation_implanted_repertoires:
            return 0

        return len(self._mutation_implanted_repertoires[seq])

    def append_count_to_seq_occurrences(self, seq: str, count: int, repertoire_id):
        """Append new count to number of occurrences for a sequence"""
        if seq in self._mutation_implanted_repertoires:
            self._mutation_implanted_repertoires[seq] += [repertoire_id] * count
        else:
            self._mutation_implanted_repertoires[seq] = [repertoire_id] * count

    def get_occurrence_limit(self, seq: str):
        """Return occurrence limit for sequence"""
        return self._mutation_occurrence_limit[seq]

    def _legal_mutation(self, seq: str, pgen: float, repertoire_id):

        if not seq:
            return False

        if seq in self._mutation_implanted_repertoires and \
                repertoire_id in self._mutation_implanted_repertoires[seq]:
            return False

        if pgen <= 0:
            return False

        num_of_occurrences = self.get_seq_occurrences(seq)

        if seq not in self._mutation_occurrence_limit:
            self._mutation_occurrence_limit[seq] = self._draw_occurrence_limit(pgen)
        occurrence_limit = self.get_occurrence_limit(seq)

        if seq not in self._mutation_implanted_repertoires:
            return True

        return num_of_occurrences < occurrence_limit

    def _draw_occurrence_limit(self, pgen):
        if self.occurrence_limit_pgen_range:
            return self._occurrence_limit_from_range(pgen)
        else:
            return self._random_occurrence_limit()

    def _occurrence_limit_from_range(self, pgen):
        """Draw random occurrence limit with max value based on pgen from occurrence limit range"""

        if float(pgen) < float(min(self.occurrence_limit_pgen_range.keys())):
            return 1

        # finding closest pgen in occurrence limit range
        max_limit = self.occurrence_limit_pgen_range.get(pgen) or self.occurrence_limit_pgen_range[
            min(self.occurrence_limit_pgen_range.keys(), key=lambda key: abs(float(key) - float(pgen)))]

        return random.choice(range(1, max_limit))

    def _random_occurrence_limit(self):
        """Draw a random occurrence limit calculated from dataset"""
        return random.choice(self.total_sequence_occurrence)

    @staticmethod
    def _total_sequence_occurrence(repertoire_dataset):
        """
        Args:
            repertoire_dataset (Dataset): Repertoire dataset

        Returns:
            dict: Dictionary with count of how many of each duplicate count there are in the dataset
        """

        # count of how often each sequence appears
        sequence_occurrence_counter = defaultdict(lambda: 0)

        for repertoire in repertoire_dataset.get_data():
            for seq, count in zip(repertoire.get_sequence_aas(), repertoire.get_counts()):
                sequence_occurrence_counter[seq] = sequence_occurrence_counter[seq] + count

        return list(sequence_occurrence_counter.values())

    @staticmethod
    def get_default_model_name(dataset: RepertoireDataset):
        """Extract generative model name (organism + chain, e.g. 'humanTRB') from dataset"""

        # TODO check if set(get_attribute("chains")) contains only one chain type

        # TODO move get_model_name to some util-thing
        valid_chain_types = ("TRB", "TRA", "IGH", "IGL", "IGK")

        dataset_organisms = dataset.get_repertoire(index=0).get_attribute("organism")
        if dataset_organisms:
            organism = dataset_organisms[0]
        else:
            organism = "human"

        chain_type = dataset.get_repertoire(index=0).get_v_genes()[0][:3]

        if chain_type not in valid_chain_types:
            raise Exception(f"No OLGA model with chain type: {chain_type}")

        return organism + chain_type
