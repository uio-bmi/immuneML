from dataclasses import dataclass
from typing import List, Union

from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.SimConfigItem import SimConfigItem
from immuneML.simulation.simulation_strategy.SimulationStrategy import SimulationStrategy


@dataclass
class SimConfig:
    """
    The simulation config defines all parameters of the simulation.
    It can contain one or more simulation config items, which define groups of repertoires or receptors
    that have the same simulation parameters, such as signals, generative model, clonal frequencies, and noise parameters.


    **Specification arguments:**

    - sim_items (dict): a list of SimConfigItems defining individual units of simulation

    - is_repertoire (bool): whether the simulation is on a repertoire (person) or sequence/receptor level

    - paired: if the simulation should output paired data, this parameter should contain a list of a list of sim_item pairs referenced by name that should be combined; if paired data is not needed, then it should be False

    - sequence_type (str): either amino_acid or nucleotide

    - simulation_strategy (str): either RejectionSampling or Implanting, see the tutorials for more information on choosing one of these

    - keep_p_gen_dist (bool): if possible, whether to keep the distribution of generation probabilities of the sequences the same as provided by the model without any signals

    - p_gen_bin_count (int): if keep_p_gen_dist is true, how many bins to use to approximate the generation probability distribution

    - remove_seqs_with_signals (bool): if true, it explicitly controls the proportions of signals in sequences and removes any accidental occurrences

    - species (str): species that the sequences come from; used to select correct genes to export full length sequences; default is 'human'

    - implanting_scaling_factor (int): determines in how many receptors to implant the signal in reach iteration; this is computed as number_of_receptors_needed_for_signal * implanting_scaling_factor; useful when using Implanting simulation strategy in combination with importance sampling, since the generation probability of some receptors with implanted signals might be very rare and those receptors might end up not being kept often with importance sampling; this parameter is only used when keep_p_gen_dist is set to True


    **YAML specification:**

    .. indent-with-spaces
    .. code-block:: yaml

        definitions:
            simulations:
                sim1:
                    is_repertoire: false
                    paired: false
                    sequence_type: amino_acid
                    simulation_strategy: RejectionSampling
                    sim_items:
                        sim_item1: # group of sequences with same simulation params
                            generative_model:
                                chain: beta
                                default_model_name: humanTRB
                                model_path: null
                                type: OLGA
                            number_of_examples: 100
                            seed: 1002
                            signals:
                                signal1: 1

    """
    sim_items: List[SimConfigItem] = None
    identifier: str = None
    is_repertoire: bool = None
    paired: Union[bool, List[List[str]]] = None
    sequence_type: SequenceType = None
    simulation_strategy: SimulationStrategy = None
    p_gen_bin_count: int = None
    keep_p_gen_dist: bool = None
    remove_seqs_with_signals: bool = None
    species: str = None
    implanting_scaling_factor: int = None

    def __str__(self):
        return ",\n".join(str(simulation_item) for simulation_item in self.sim_items)

    def get_total_seq_count_for_signal(self, signal_id: str, model_name: str) -> int:
        sim_item_names = model_name.split("__")
        total_count = 0
        for sim_item_name in sim_item_names:
            total_count += self._get_seq_count_for_sim_item(signal_id, sim_item_name)
        return round(total_count)

    def get_total_seq_count(self, model_name: str) -> int:
        sim_item_names = model_name.split("__")
        sim_items = [sim_item for sim_item in self.sim_items if sim_item.name in sim_item_names]
        return round(
            sum([sim_item.number_of_examples * (sim_item.receptors_in_repertoire_count if self.is_repertoire else 1)
                 for sim_item in sim_items]))

    def _get_seq_count_for_sim_item(self, signal_id: str, sim_item_name: str) -> int:
        count = 0
        sim_item = [sim_item for sim_item in self.sim_items if sim_item.name == sim_item_name][0]
        signal_with_proportions = {k: v for k, v in sim_item.signal_proportions.items()
                                   if k.id == signal_id or (signal_id in k.id and "__" in k.id)}
        if signal_with_proportions:
            count = sim_item.number_of_examples * sum(signal_with_proportions.values())
            if self.is_repertoire:
                count *= sim_item.receptors_in_repertoire_count
        return count