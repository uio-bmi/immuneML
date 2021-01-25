class HPLabelState:

    def __init__(self, label, auxiliary_labels):
        self.label = label
        self.auxiliary_labels = auxiliary_labels

        # computed
        self.assessment_items = {}
        self.selection_state = None

    @property
    def optimal_assessment_item(self):
        return self.assessment_items[str(self.selection_state.optimal_hp_setting)]

    @property
    def optimal_hp_setting(self):
        return self.selection_state.optimal_hp_setting
