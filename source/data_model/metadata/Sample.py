# quality: gold


class Sample:

    def __init__(self, identifier, name: str = None, custom_params: dict = None):
        self.id = identifier
        self.name = name
        self.custom_params = custom_params
