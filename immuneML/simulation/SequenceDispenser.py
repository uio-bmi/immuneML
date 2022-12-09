import random
from collections import defaultdict

import pandas as pd

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.generative_models.OLGA import OLGA
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation


class SequenceDispenser:
    # TODO handle not finding mutations better
    # TODO improve assert messages
    # TODO improve variable names

    """
    TODO: ERROR!!!
        - All signals only have 1 signal_id in annotation
        - Only 1 motif gets implanted of the signals???
    """

    def __init__(self, sequence_count: dict):

        assert sequence_count, "Sequence count has not been set"
        self.sequence_count = sequence_count

        self.seeds = []

        # dict with (key, value) = (mutation seq., list of repertoire ids.)
        self._mutation_implanted_repertoires = {}
        self._mutation_count_limit = {}

        self.olga = OLGA.build_object(model_path=None, default_model_name="humanTRB", chain=None,
                                      use_only_productive=False)
        self.olga.load_model()

        self.amino_alphabet = EnvironmentSettings.get_sequence_alphabet(sequence_type=SequenceType.AMINO_ACID)

    def add_seed_sequence(self, motif: Motif):
        if motif not in self.seeds:
            self.seeds.append(motif)

    def generate_mutation(self, repertoire_id) -> Motif:
        assert self.seeds, "No signals to mutate"

        # pick random seed sequence from given signal
        seed_motif = random.choice(self.seeds)

        mutation_position_probabilities = seed_motif.mutation_position_possibilities

        #        if mutation_position_probabilities is not None:
        #            mutation_position_probabilities = {key: float(value) for key, value in
        #                                              mutation_position_probabilities.items()}
        #            assert all(isinstance(key, int) for key in mutation_position_probabilities.keys()) \
        #                   and all(isinstance(val, float) for val in mutation_position_probabilities.values()) \
        #                   and 0.99 <= sum(mutation_position_probabilities.values()) <= 1, \
        #                f"For each possible mutation position a probability between 0 and 1 has to be assigned so that the " \
        #                f"probabilities for all distance possibilities sum to 1. sum({mutation_position_probabilities.values()}) = {sum(mutation_position_probabilities.values())}"

        seed_seq = seed_motif.seed
        v_call = seed_motif.v_call
        j_call = seed_motif.j_call

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
                raise Exception(f"Could not find valid mutation for {seed_seq}")
                # print(f"Could not find valid mutation for {seed_seq}")
                # return self.generate_mutation(repertoire_id)

        print(f"\nFINAL PGEN: {pgen}\n"
              f"MUTATION: {mutation}")
        print(f"{attempt_counter=}")

        if mutation in self._mutation_implanted_repertoires:
            self._mutation_implanted_repertoires[mutation].append(repertoire_id)
        else:
            self._mutation_implanted_repertoires[mutation] = [repertoire_id]

        # TODO: unique id
        return Motif(identifier="m1", instantiation=GappedKmerInstantiation(), seed=mutation, v_call=v_call,
                     j_call=j_call)

    def _mutate_sequence(self, seq: str, mutation_position_probabilities: dict):
        mutation_position = random.choices(list(mutation_position_probabilities.keys()),
                                           weights=list(mutation_position_probabilities.values()), k=1)[0]
        amino_acid = random.choice(self.amino_alphabet)

        mutation = seq[:mutation_position] + amino_acid + seq[mutation_position + 1:]

        return mutation

    def _get_seq_count(self, seq: str):
        if seq not in self._mutation_implanted_repertoires:
            return 0

        return len(self._mutation_implanted_repertoires[seq])

    def _legal_mutation(self, seq: str, pgen: float, repertoire_id):

        if seq in self._mutation_implanted_repertoires and \
                repertoire_id in self._mutation_implanted_repertoires[seq]:
            return False

        if pgen <= 0:
            return False

        if seq not in self._mutation_implanted_repertoires:
            return True

        count = self._get_seq_count(seq)

        if seq not in self._mutation_count_limit:
            self._mutation_count_limit[seq] = self._draw_random_limit_count()
        count_limit = self._mutation_count_limit[seq]

        print(f"------------------------{count_limit=}")

        return count < count_limit

    def _draw_random_limit_count(self):
        random_count = \
            random.choices(list(self.sequence_count.keys()), weights=list(self.sequence_count.values()), k=1)[0]

        # TODO decide good variance
        variance = random_count / 1.2
        limit = round(random.gauss(random_count, variance))

        if limit < 1:
            limit = 1

        return limit

    @staticmethod
    def total_sequence_count(repertoire_dataset):

        sequence_count = defaultdict(lambda: 0)

        for repertoire in repertoire_dataset:
            for seq, count in zip(repertoire.get_sequence_aas(), repertoire.get_counts()):
                sequence_count[seq] = sequence_count[seq] + count

        count_count = defaultdict(lambda: 0)

        for _, v in sequence_count.items():
            count_count[v] = count_count[v] + 1

        return count_count
