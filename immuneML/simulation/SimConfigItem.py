from dataclasses import dataclass, field
from typing import List, Dict, Union

from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.simulation.implants.Signal import Signal, SignalPair


@dataclass
class SimConfigItem:
    """
    When performing a simulation, one or more simulation config items can be specified. Config items define groups of
    repertoires or receptors that have the same simulation parameters, such as signals, generative model, clonal
    frequencies, noise parameters.


    Specification arguments:

    - signals (dict): signals for the simulation item and the proportion of sequences in the repertoire that will have the given signal. For receptor-level simulation, the proportion will always be 1.

    - is_noise (bool): indicates whether the implanting should be regarded as noise; if it is True, the signals will be implanted as specified, but the repertoire/receptor in question will have negative class.

    - generative_model: parameters of the generative model, including its type, path to the model; currently supported models are OLGA and ExperimentalImport

    - seed (int): starting random seed for the generative model (it should differ across simulation items, or it can be set to null when not used)

    - false_positives_prob_in_receptors (float): when performing repertoire level simulation, what percentage of sequences should be false positives

    - false_negative_prob_in_receptors (float): when performing repertoire level simulation, what percentage of sequences should be false negatives

    - immune_events (dict): a set of key-value pairs that will be added to the metadata (same values for all data generated in one simulation sim_item) and can be later used as labels

    - default_clonal_frequency (dict): clonal frequency in Ligo is simulated through `scipy's zeta distribution function for generating random numbers <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zipf.html>`_, with parameters provided under default_clonal_frequency parameter. These parameters will be used to assign count values to sequences that do not contain any signals if they are required by the simulation. If clonal frequency shouldn't be used, this parameter can be None

    .. indent with spaces
    .. code-block:: yaml

        clonal_frequency:
            a: 2 # shape parameter of the distribution
            loc: 0 # 0 by default but can be used to shift the distribution

    - sequence_len_limits (dict): allows for filtering the generated sequences by length, needs to have parameters min and max specified; if not used, min/max should be -1

    .. indent with spaces
    .. code-block:: yaml

        sequence_len_limits:
            min: 4 # keep sequences of length 4 and longer
            max: -1 # no limit on the max length of the sequences

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        simulations: # definitions of simulations should be under key simulations in the definitions part of the specification
            # one simulation with multiple implanting objects, a part of definition section
            my_simulation:
                sim_item1:
                    number_of_examples: 10
                    seed: null # don't use seed
                    receptors_in_repertoire_count: 100
                    generative_model:
                        chain: beta
                        default_model_name: humanTRB
                        model_path: null
                        type: OLGA
                    signals:
                        my_signal: 0.25
                        my_signal2: 0.01
                        my_signal__my_signal2: 0.02 # my_signal and my_signal2 will co-occur in 2% of the receptors in all 10 repertoires
                sim_item2:
                    number_of_examples: 5
                    receptors_in_repertoire_count: 150
                    seed: 10 #
                    generative_model:
                        chain: beta
                        default_model_name: humanTRB
                        model_path: null
                        type: OLGA
                    signals:
                        my_signal: 0.75
                    default_clonal_frequency:
                        a: 2
                    sequence_len_limits:
                        min: 3


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
    default_clonal_frequency: dict = None
    sequence_len_limits: dict = None

    @property
    def signals(self) -> List[Signal]:
        return list(self.signal_proportions.keys())
