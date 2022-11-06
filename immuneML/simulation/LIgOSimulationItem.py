from dataclasses import dataclass, field
from typing import List

from immuneML.simulation.generative_models.GenerativeModel import GenerativeModel
from immuneML.simulation.implants.Signal import Signal


@dataclass
class LIgOSimulationItem:
    """
    When performing a Simulation, one or more implantings can be specified. An implanting represents
    a set of signals which are implanted in a RepertoireDataset with given rates.

    Multiple implantings may be specified in one simulation. In this case, each implanting will only
    affect its own partition of the dataset, so each repertoire can only receive implanted signals from
    one implanting. This way, implantings can be used to ensure signals do not overlap (one implanting per
    signal), or to ensure signals always occur together (multiple signals per implanting).


    Arguments:

        signals (list): The list of :ref:`Signal` objects to be implanted in a subset of the repertoires in a RepertoireDataset.
        When multiple signals are specified, this means that all of these signals are implanted in
        the same repertoires in a RepertoireDataset, although they may not be implanted in the same sequences
        within those repertoires (this depends on the :py:obj:`~immuneML.simulation.signal_implanting.SignalImplantingStrategy.SignalImplantingStrategy`).

        repertoire_implanting_rate (float): The proportion of sequences in a Repertoire where a motif associated
        with one of the signals should be implanted.

        is_noise (bool): indicates whether the implanting should be regarded as noise; if it is True, the signals will be implanted as specified, but
        the repertoire/receptor in question will have negative class.

        generative_model: parameters of the generative model, including its type, path to the model

        seed (int): starting random seed for the generative model (ideally should differ across simulation items)

        immune_events (dict): a set of key-value pairs that will be added to the metadata (same values for all data generated in one simulation item) and can be later used as labels

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        simulations: # definitions of simulations should be under key simulations in the definitions part of the specification
            # one simulation with multiple implanting objects, a part of definition section
            my_simulation:
                my_implanting_1:
                    signals:
                        - my_signal
                    dataset_implanting_rate: 0.5
                    repertoire_implanting_rate: 0.25
                my_implanting_2:
                    signals:
                        - my_signal
                    dataset_implanting_rate: 0.2
                    repertoire_implanting_rate: 0.75


    """

    signals: List[Signal]
    name: str = ""
    repertoire_implanting_rate: float = None
    is_noise: bool = False
    seed: int = None
    generative_model: GenerativeModel = None
    number_of_examples: int = 0
    receptors_in_repertoire_count: int = 0
    immune_events: dict = field(default_factory=dict)

    def __str__(self):
        return str(vars(self))