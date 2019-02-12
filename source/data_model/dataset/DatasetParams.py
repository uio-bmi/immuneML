# quality: gold


class DatasetParams:

    def __init__(self, number_of_examples: int = None,
                 number_of_sequences: int = None,
                 path=None,
                 encoding: str = None,
                 labels: dict = None,
                 sample_param_names: list = None):
        self.number_of_examples = number_of_examples
        self.number_of_sequences = number_of_sequences
        self.path = path
        self.encoding = encoding
        self.labels = labels if labels is not None else {}
        self.sample_param_names = [name.lower() for name in sample_param_names] if sample_param_names is not None else []

    def get_sample_params(self):
        return self.sample_param_names

    def set_encoding(self, encoding: str):
        self.encoding = encoding

    def add_label(self, name: str, value):
        if name not in self.labels:
            self.labels[name] = [value]
        elif value not in self.labels[name]:
            self.labels[name].append(value)

    def __str__(self):
        description = "examples: " + str(self.number_of_examples) \
                        + ", sequences: " + str(self.number_of_sequences)
