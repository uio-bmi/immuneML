class Label:

    def __init__(self, name: str, values: list = None, auxiliary_label_names: list = None):
        self.name = name
        self.values = values
        self.auxiliary_label_names = auxiliary_label_names
