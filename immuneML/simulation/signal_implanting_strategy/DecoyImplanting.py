import copy
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import List

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceAnnotation import SequenceAnnotation
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.simulation.SimulationState import SimulationState
from immuneML.simulation.generative_models.OLGA import OLGA
from immuneML.simulation.implants.ImplantAnnotation import ImplantAnnotation
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation
from immuneML.simulation.sequence_implanting.SequenceImplantingStrategy import SequenceImplantingStrategy
from immuneML.simulation.signal_implanting_strategy import ImplantingComputation
from immuneML.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy


class DecoyImplanting(SignalImplantingStrategy):
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
                implanting: FullSequence

    """

    default_model_name = None

    def __init__(self, implanting: SequenceImplantingStrategy = None, sequence_position_weights: dict = None,
                 implanting_computation: ImplantingComputation = None, mutation_hamming_distance: int = 1,
                 occurrence_limit_pgen_range: dict = None, overwrite_sequences: bool = False,
                 nr_of_decoys: int = None, dataset_implanting_rate_per_decoy: float = None,
                 repertoire_implanting_rate_per_decoy: float = None):
        super().__init__(implanting, sequence_position_weights, implanting_computation, mutation_hamming_distance,
                         occurrence_limit_pgen_range, overwrite_sequences, nr_of_decoys,
                         dataset_implanting_rate_per_decoy, repertoire_implanting_rate_per_decoy)

        self.decoy_sequences = None

    def implant_in_repertoire(self, repertoire: Repertoire, repertoire_implanting_rate: float, signal, path: Path):

        assert all("/" not in motif.seed for motif in signal.motifs), \
            f'DecoyImplanting: motifs cannot include gaps. Check motifs {[motif.identifier for motif in signal.motifs]}.'

        sequences = repertoire.sequences
        new_sequence_count = math.ceil(len(sequences) * repertoire_implanting_rate)
        assert new_sequence_count > 0, \
            f"DecoyImplanting: there are too few sequences ({len(sequences)}) in the repertoire with identifier {repertoire.identifier} " \
            f"to have the given repertoire implanting rate ({repertoire_implanting_rate}). Please consider increasing the repertoire implanting rate."
        new_sequences = self._create_new_sequences(sequences, new_sequence_count, signal)
        metadata = copy.deepcopy(repertoire.metadata)
        metadata[signal.id] = True

        return Repertoire.build_from_sequence_objects(new_sequences, path, metadata)

    def _create_new_sequences(self, sequences, new_sequence_count, signal) -> List[ReceptorSequence]:

        decoys = self._get_decoys_to_implant(sequences)

        if self.overwrite_sequences:
            random.shuffle(sequences)
            new_sequences = sequences[:-new_sequence_count + len(decoys)]
        else:
            new_sequences = sequences

        seed_counts = defaultdict(lambda: 0)

        for _ in range(new_sequence_count):
            seed_counts[signal.motifs.index(random.choice(signal.motifs))] += 1

        for seed_index, count in seed_counts.items():
            seed = signal.motifs[seed_index]

            seed_instance = seed.instantiate_motif()
            annotation = SequenceAnnotation([ImplantAnnotation(signal_id=signal.id, motif_id=seed.identifier,
                                                               motif_instance=seed_instance.instance, position=0)])
            metadata = SequenceMetadata(v_gene=seed.v_call, j_gene=seed.j_call,
                                        count=count,
                                        # TODO get chain and account for maybe no chain. use repertoire.get_chains()[0],
                                        chain="B",
                                        region_type=sequences[0].metadata.region_type)

            new_sequences.append(
                ReceptorSequence(amino_acid_sequence=seed_instance.instance, annotation=annotation, metadata=metadata))

        new_sequences += decoys

        return new_sequences

    def implant_in_receptor(self, receptor, signal, is_noise: bool):
        raise RuntimeError("DecoyImplanting was called on a receptor object. Check the simulation parameters.")

    def _get_decoys_to_implant(self, sequences):

        if self.decoy_sequences is None:

            assert self.default_model_name, "Default model name for generative model has not been set"

            olga = OLGA.build_object(model_path=None, default_model_name=self.default_model_name, chain=None,
                                     use_only_productive=False)
            olga.load_model()

            nr_of_decoys = self.nr_of_decoys

            self.decoy_sequences = olga.generate_sequence_dataframe(nr_of_decoys)

        decoys_to_implant_in_current_repertoire = []

        for index, row in self.decoy_sequences.iterrows():
            if random.random() < self.dataset_implanting_rate_per_decoy:
                decoy = Motif(identifier=f"decoy_{row['sequence_aa']}", instantiation=GappedKmerInstantiation(),
                              seed=row["sequence_aa"],
                              v_call=row["v_call"],
                              j_call=row["j_call"])

                count = int(self.repertoire_implanting_rate_per_decoy * len(sequences))
                variance = count / 2

                decoy_implant_count = round(random.gauss(count, variance))

                if decoy_implant_count < 1:
                    decoy_implant_count = 1

                decoy_instance = decoy.instantiate_motif()
                annotation = SequenceAnnotation(
                    [ImplantAnnotation(signal_id=DecoyImplanting.__name__, motif_id=decoy.identifier,
                                       motif_instance=decoy_instance.instance, position=0)])
                metadata = SequenceMetadata(v_gene=decoy.v_call, j_gene=decoy.j_call,
                                            count=decoy_implant_count,
                                            # TODO get chain and account for maybe no chain. use repertoire.get_chains()[0],
                                            chain="B",
                                            region_type=sequences[0].metadata.region_type)

                decoys_to_implant_in_current_repertoire.append(
                    ReceptorSequence(amino_acid_sequence=decoy_instance.instance, annotation=annotation,
                                     metadata=metadata))

        return decoys_to_implant_in_current_repertoire

    def _implant_decoys(self, repertoire: Repertoire, decoys):

        if self.overwrite_sequences:
            new_sequences = repertoire.sequences[:-len(decoys)]
        else:
            new_sequences = repertoire.sequences

        return new_sequences + decoys

    @staticmethod
    def implant_decoys_in_repertoire(repertoire: Repertoire, simulation_state: SimulationState, metadata):

        decoy_implanting_strategy = [signal.implanting_strategy for implanting in
                                     simulation_state.simulation.implantings for signal in
                                     implanting.signals if isinstance(signal.implanting_strategy, DecoyImplanting)][0]

        # TODO figure out what to do with multiple decoy implantings
        #  For now, only account for one decoy

        path = simulation_state.result_path / "repertoires/"

        decoys = decoy_implanting_strategy._get_decoys_to_implant(repertoire.sequences)

        new_sequences = decoy_implanting_strategy._implant_decoys(repertoire, decoys)

        return Repertoire.build_from_sequence_objects(new_sequences, path, metadata)
