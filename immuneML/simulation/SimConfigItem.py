from dataclasses import dataclass, field
from typing import List, Dict, Union

from immuneML.simulation.generative_models.GenerativeModel import GenerativeModel
from immuneML.simulation.implants.Signal import Signal, SignalPair


@dataclass
class SimConfigItem:
    """
    When performing a SimConfig, one or more implantings can be specified. An implanting represents
    a set of signals which are implanted in a RepertoireDataset with given rates.

    Multiple implantings may be specified in one simulation. In this case, each implanting will only
    affect its own partition of the dataset, so each repertoire can only receive implanted signals from
    one implanting. This way, implantings can be used to ensure signals do not overlap (one implanting per
    signal), or to ensure signals always occur together (multiple signals per implanting).


    Arguments:

        signal_proportions (dict): signals for the simulation item and the proportion of sequences in the repertoire that will have the given signal. For receptor-level simulation, the proportion will always be 1.

        is_noise (bool): indicates whether the implanting should be regarded as noise; if it is True, the signals will be implanted as specified, but
        the repertoire/receptor in question will have negative class.

        generative_model: parameters of the generative model, including its type, path to the model

        seed (int): starting random seed for the generative model (ideally should differ across simulation items)

        false_positives_prob_in_receptors (float): when doing repertoire level simulation, what percentage of sequences should be false positives

        false_negative_prob_in_receptors (float): when doing repertoire level simulation, what percentage of sequences should be false negatives

        immune_events (dict): a set of key-value pairs that will be added to the metadata (same values for all data generated in one simulation sim_item) and can be later used as labels

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        simulations: # definitions of simulations should be under key simulations in the definitions part of the specification
            # one simulation with multiple implanting objects, a part of definition section
            my_simulation:
                sim_item1:
                    number_of_examples: 10
                    receptors_in_repertoire_count: 100
                    signals:
                        my_signal: 0.25
                        my_signal2: 0.01
                        my_signal__my_signal2: 0.02
                sim_item2:
                    number_of_examples: 5
                    receptors_in_repertoire_count: 150
                    signals:
                        my_signal: 0.75


    """

    signal_proportions: Dict[Union[Signal, SignalPair], float]
    name: str = ""
    is_noise: bool = False
    seed: int = None
    generative_model: GenerativeModel = None
    number_of_examples: int = 0
    receptors_in_repertoire_count: int = 0
    false_positive_prob_in_receptors: float = 0.
    false_negative_prob_in_receptors: float = 0.
    immune_events: dict = field(default_factory=dict)

    @property
    def signals(self) -> List[Signal]:
        return list(self.signal_proportions.keys())