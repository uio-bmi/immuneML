# quality: gold


class Label:
    """
    Class storing information about the label for machine learning methods;
    Only what is stored in the objects of this class can be used for supervised learning;
    Consists of only label name and value, where the value will typically be integer
    """
    def __init__(self, value, name: str = None):
        self.value = value
        self.name = name

    def get_name(self):
        return self.name

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def set_name(self, name: str):
        self.name = name
