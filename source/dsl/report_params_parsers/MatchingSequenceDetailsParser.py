from source.util.ReflectionHandler import ReflectionHandler


class MatchingSequenceDetailsParser:
    """
    The definition for MatchingSequenceDetails has the following format in the DSL:

    .. highlight:: yaml
    .. code-block:: yaml

        report1:
            type: MatchingSequenceDetails
            params:
                reference_sequences:
                    path: ./seqs.csv
                    format: VDJDB
                max_edit_distance: 2

    """

    @staticmethod
    def parse(params: dict):

        sequence_import = ReflectionHandler.get_class_by_name("{}SequenceImport".format(params["reference_sequences"]["format"]))

        return {
            "reference_sequences": sequence_import.import_items(params["reference_sequences"]["path"]),
            "max_edit_distance": params["max_edit_distance"]
        }, params
