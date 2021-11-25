class Simulation:

    def __init__(self, simulation_items: list, identifier=None):
        self.simulation_items = simulation_items
        self.identifier = identifier

    def __str__(self):
        return ",\n".join(str(simulation_item) for simulation_item in self.simulation_items)
