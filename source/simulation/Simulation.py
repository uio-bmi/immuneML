class Simulation:

    """
    Simulation object encapsulates the parameters of simulating healthy / affected repertoires
    (works only on the repertoire level)
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
