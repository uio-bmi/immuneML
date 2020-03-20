class Implanting:

    """
    When performing a Simulation, one or more implantings can be specified. An implanting represents
    a set of signals which are implanted in a RepertoireDataset with given rates.

    Multiple implantings can be specified in one Simulation. In this case, each implanting will only
    affect its own partition of the dataset, so each repertoire can only receive implanted signals from
    one implanting. This way, implantings can be used to ensure signals do not overlap (one implanting per
    signal), or to ensure signals always occur together (multiple signals per implanting).


    Arguments:
        signals (list): The list of :py:obj:`~source.simulation.implants.Signal.Signal` objects to be implanted
            in a subset of the repertoires in of a RepertoireDataset.
            When multiple signals are specified, this means that all of these signals are implanted in
            the same repertoires in a RepertoireDataset, although they may not be implanted in the same sequences
            within those repertoires (this depends on the :py:obj:`~source.simulation.signal_implanting_strategy.SignalImplantingStrategy.SignalImplantingStrategy`).
        dataset_implanting_rate (float): The proportion of repertoires in the RepertoireDataset in which the
            signals should be implanted. When specifying multiple implantings, the sum of all dataset_implanting_rates
            should not exceed 1.
        repertoire_implanting_rate (float): The proportion of sequences in a Repertoire where a motif associated
            with one of the signals should be implanted.


    Specification:

        motifs:
            my_motif:
                ...

        signals:
            my_signal:
                motifs:
                    - my_motif
                    - ...
                implanting: HealthySequence
                ...

        simulation:
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

    def __init__(self, dataset_implanting_rate: float, repertoire_implanting_rate: float, signals: list, name: str = ""):
        self.dataset_implanting_rate = dataset_implanting_rate
        self.repertoire_implanting_rate = repertoire_implanting_rate
        self.signals = signals
        self.name = name

    def __str__(self):
        return self.name + ": dataset_implanting_rate: {}, " \
                           "repertoire_implanting_rate: {}, " \
                           "signals: {}".format(self.dataset_implanting_rate,
                                                self.repertoire_implanting_rate,
                                                [str(s) for s in self.signals])
