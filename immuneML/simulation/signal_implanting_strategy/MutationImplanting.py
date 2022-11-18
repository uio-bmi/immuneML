import copy
import math
import random
from pathlib import Path
from typing import List

import pandas as pd

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceAnnotation import SequenceAnnotation
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.generative_models.OLGA import OLGA
from immuneML.simulation.implants.ImplantAnnotation import ImplantAnnotation
from immuneML.simulation.implants.MotifInstance import MotifInstance
from immuneML.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy


class MutationImplanting(SignalImplantingStrategy):
    """
    This class represents a :py:obj:`~immuneML.simulation.signal_implanting_strategy.SignalImplantingStrategy.SignalImplantingStrategy`
    where signals will be implanted in the repertoire by replacing `repertoire_implanting_rate` percent of the sequences with sequences
    generated from the motifs of the signal. Motifs here cannot include gaps and the motif instances are the full sequences and will be
    a part of the repertoire.

    Note: when providing the sequence to be implanted, check if the import setting regarding the sequence type (CDR3 vs IMGT junction) matches
    the sequence to be implanted. For example, if importing would convert junction sequences to CDR3, but the sequence specified here for implanting
    would be the junction, the results of the simulation could be inconsistent.

    Arguments: this signal implanting strategy has no arguments.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        motifs:
            my_motif: # cannot include gaps
                ...

        signals:
            my_signal:
                motifs:
                    - my_motif
                implanting: Mutation

    """

    def implant_in_repertoire(self, repertoire: Repertoire, repertoire_implanting_rate: float, signal, path: Path):

        assert all("/" not in motif.seed for motif in signal.motifs), \
            f'MutationImplanting: motifs cannot include gaps. Check motifs {[motif.identifier for motif in signal.motifs]}.'

        sequences = repertoire.sequences

        new_sequence_count = math.ceil(len(sequences) * repertoire_implanting_rate)
        assert new_sequence_count > 0, \
            f"MutationImplanting: there are too few sequences ({len(sequences)}) in the repertoire with identifier {repertoire.identifier} " \
            f"to have the given repertoire implanting rate ({repertoire_implanting_rate}). Please consider increasing the repertoire implanting rate."
        new_sequences = self._create_new_sequences(sequences, new_sequence_count, signal)
        metadata = copy.deepcopy(repertoire.metadata)
        metadata[signal.id] = True

        for i in sequences:
            i.metadata.custom_params = {}

        return Repertoire.build_from_sequence_objects(new_sequences, path, metadata)

    def _create_new_sequences(self, sequences, new_sequence_count, signal) -> List[ReceptorSequence]:
        new_sequences = sequences[:-new_sequence_count]

        # V and J genes for when there is none from the seed sequence
        placeholder_v_gene = None # "TRBV5-1"
        placeholder_j_gene = None # "TRBJ4-2"

        all_mutations = []

        # TODO keep track of which mutations are for which signals so the implant annotation is correct
        for motif in signal.motifs:
            all_mutations += self._get_mutations(motif.seed, placeholder_v_gene, placeholder_j_gene)

        # TODO weighted choices by pgen? (random.choices(weights=[p1,p2,p3,...] or maybe cumulative weights with mutations ordered by pgen)
        sequences_to_implant = random.sample(all_mutations, new_sequence_count)

        for index, seq in enumerate(sequences_to_implant):
            motif_instance = MotifInstance(instance=seq, gap=0)

            annotation = SequenceAnnotation([ImplantAnnotation(signal_id=signal.id, motif_id=f"mutation{index}",
                                                               motif_instance=motif_instance, position=0)])
            metadata = SequenceMetadata(v_gene=placeholder_v_gene, j_gene=placeholder_j_gene, count=1, chain="B",
                                        region_type=sequences[0].metadata.region_type)
            new_sequences.append(
                ReceptorSequence(amino_acid_sequence=motif_instance.instance, annotation=annotation, metadata=metadata))

        return new_sequences

    def _get_mutations(self, seed_seq: str, v_gene: str, j_gene: str):
        mutations = []
        mutation_pos = []
        amino_alphabet = EnvironmentSettings.get_sequence_alphabet(sequence_type=SequenceType.AMINO_ACID)

        # TODO test different values (maybe add as input variable?)
        mutation_range_from_ends = 2
        print(f"Generating mutations from sequence {seed_seq} from position with indexes {mutation_range_from_ends} to "
              f"{len(seed_seq) - mutation_range_from_ends} (exclusionary). Area to be mutated: {seed_seq[mutation_range_from_ends: len(seed_seq) - mutation_range_from_ends]}")

        for i in range(mutation_range_from_ends, len(seed_seq) - mutation_range_from_ends):
            for amino_acid in amino_alphabet:
                mutation = seed_seq[:i] + amino_acid + seed_seq[i + 1:]

                if mutation == seed_seq:
                    continue

                mutations.append(mutation)
                mutation_pos.append(i)

        olga = OLGA.build_object(model_path=None, default_model_name="humanTRB", chain=None, use_only_productive=False)
        olga.load_model()

        df = pd.DataFrame(data={
            "sequence_aas": mutations,
            "v_genes": v_gene,
            "j_genes": j_gene,
            "mutation_pos": mutation_pos
        })

        df.drop_duplicates(subset=["sequence_aas"], inplace=True, ignore_index=True)

        # compute pgen
        df["pgen"] = olga.compute_p_gens(df, SequenceType.AMINO_ACID)

        df = df[df['pgen'] > 0].reset_index(drop=True)

        print(f"\nNr. of possible seqs.: {len(df)}")

        print("\nMEAN ___________________")
        print(df.groupby('mutation_pos').mean())
        print()

        return df["sequence_aas"].tolist()

    def implant_in_receptor(self, receptor, signal, is_noise: bool):
        raise RuntimeError("MutationImplanting was called on a receptor object. Check the simulation parameters.")
