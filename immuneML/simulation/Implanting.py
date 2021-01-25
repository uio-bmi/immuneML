from typing import List

from immuneML.simulation.implants.Signal import Signal


class Implanting:

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
        within those repertoires (this depends on the :py:obj:`~immuneML.simulation.signal_implanting_strategy.SignalImplantingStrategy.SignalImplantingStrategy`).

        dataset_implanting_rate (float): The proportion of repertoires in the RepertoireDataset in which the
        signals should be implanted. When specifying multiple implantings, the sum of all dataset_implanting_rates
        should not exceed 1.

        repertoire_implanting_rate (float): The proportion of sequences in a Repertoire where a motif associated
        with one of the signals should be implanted.

        is_noise (bool): indicates whether the implanting should be regarded as noise; if it is True, the signals will be implanted as specified, but
        the repertoire/receptor in question will have negative class.


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

            # a simulation where the signals is present in the negative class as well (e.g. wrong labels, or by chance)
            noisy_simulation:
                positive_class_implanting:
                    signals:
                        - my_signal
                    dataset_implanting_rate: 0.5
                    repertoire_implanting_rate: 0.1 # 10% of the repertoire includes the signal in the positive class
                negative_class_implanting:
                    signals:
                        - my_signal
                    is_noise: True # means that signal will be implanted, but the label will have negative class
                    dataset_implanting_rate: 0.5
                    repertoire_implanting_rate: 0.01 # 1% of negative class repertoires has the signal

            # in case of defining implanting for paired chain immune receptor data the simulation with implanting objects would be:
            my_receptor_simulation:
                my_receptor_implanting_1: # repertoire_implanting_rate is omitted in this case, as it is not applicable
                    signals:
                        - my_receptor_signal
                    dataset_implanting_rate: 0.4 # 40% of the receptors will have signal my_receptor_signal implanted and 60% will not

    """

    def __init__(self, dataset_implanting_rate: float, signals: List[Signal], name: str = "", repertoire_implanting_rate: float = None,
                 is_noise: bool = False):
        self.dataset_implanting_rate = dataset_implanting_rate
        self.repertoire_implanting_rate = repertoire_implanting_rate
        self.signals = signals
        self.is_noise = is_noise
        self.name = name

    def __str__(self):
        return self.name + ":\n dataset_implanting_rate: {}, \n" \
                           "repertoire_implanting_rate: {}, \n" \
                           "signals: {}".format(self.dataset_implanting_rate,
                                                self.repertoire_implanting_rate,
                                                str([str(s) for s in self.signals])[1:-1])
