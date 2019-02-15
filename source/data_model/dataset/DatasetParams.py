# quality: gold


class DatasetParams:

    def __init__(self, sample_param_names: list = None):
        self.sample_param_names = [name.lower() for name in sample_param_names] if sample_param_names is not None else []

    def get_sample_params(self):
        return self.sample_param_names
