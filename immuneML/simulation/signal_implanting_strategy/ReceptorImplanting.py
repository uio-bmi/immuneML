import random

from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy


class ReceptorImplanting(SignalImplantingStrategy):
    """
    This class represents a :py:obj:`~immuneML.simulation.signal_implanting_strategy.SignalImplantingStrategy.SignalImplantingStrategy`
    where signals will be implanted in both chains of immune receptors. This class should be used only when simulating paired chain data.

    Arguments:

        implanting: name of the implanting strategy, here Receptor

        sequence_position_weights (dict): A dictionary describing the relative weights for implanting a signal at each given IMGT position in the receptor sequence. If sequence_position_weights are not set, then SequenceImplantingStrategy will make all of the positions equally likely for each receptor sequence.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        motifs:
            my_motif:
                ...

        signals:
            my_signal:
                motifs:
                    - my_motif
                    - ...
                implanting: Receptor
                sequence_position_weights:
                    109: 1
                    110: 2
                    111: 5
                    112: 1

    """

    def implant_in_receptor(self, receptor, signal, is_noise: bool):
        new_receptor = receptor.clone()

        motif = random.choice(signal.motifs)

        sequence1 = self.implant_in_sequence(getattr(new_receptor, motif.name_chain1.name.lower()), signal, motif, motif.name_chain1)
        sequence2 = self.implant_in_sequence(getattr(new_receptor, motif.name_chain2.name.lower()), signal, motif, motif.name_chain2)

        setattr(new_receptor, motif.name_chain1.name.lower(), sequence1)
        setattr(new_receptor, motif.name_chain2.name.lower(), sequence2)

        new_receptor.metadata[f"signal_{signal.id}"] = True if not is_noise else False

        return new_receptor

    def implant_in_repertoire(self, repertoire: Repertoire, repertoire_implanting_rate: float, signal, path):
        raise RuntimeError("ReceptorImplanting was called on a repertoire object. Check the simulation parameters.")
