import os

from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.IO.dataset_import.IRISSequenceImport import IRISSequenceImport
from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.ImportHelper import ImportHelper
from source.util.ParameterValidator import ParameterValidator
from source.util.ReflectionHandler import ReflectionHandler


class MatchedReferenceUtil:
    '''
    Utility class for MatchedChainEncoder and MatchedReceptorsEncoder
    '''

    @staticmethod
    def prepare_reference_parameters(reference_params: dict, location: str, paired: bool):
        ParameterValidator.assert_keys(list(reference_params.keys()), ["format", "params"], location,
                                       "reference")

        seq_import_params = reference_params["params"] if "params" in reference_params else {}

        assert os.path.isfile(seq_import_params["path"]), f"{location}: the file {seq_import_params['path']} does not exist. " \
                                                  f"Specify the correct path under reference."

        if "paired" in seq_import_params:
            assert seq_import_params["paired"] == paired, f"{location}: paired must be {paired} for SequenceImport"
        else:
            seq_import_params["paired"] = paired

        format_str = reference_params["format"]


        if format_str == "IRIS":
            receptors = IRISSequenceImport.import_items(**seq_import_params)
        else: # todo refactor this whole part...........
            import_class = ReflectionHandler.get_class_by_name("{}Import".format(format_str))
            params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path + "datasets/",
                                              DefaultParamsLoader._convert_to_snake_case(format_str))
            for key, value in seq_import_params.items():
                params[key] = value
            params["paired"] = paired

            processed_params = DatasetImportParams.build_object(**params)

            receptors = ImportHelper.import_items(import_class, reference_params["path"], processed_params)



        return receptors


#
#
#
#
#
# in sequence class:
#
#
#         ParameterValidator.assert_keys(list(reference_sequences.keys()), ["format", "path"], location, "reference_sequences")
#         ParameterValidator.assert_in_valid_list(summary.upper(), [item.name for item in SequenceMatchingSummaryType], location, "summary")
#
#         # valid_formats = ReflectionHandler.discover_classes_by_partial_name("SequenceImport", "sequence_import/")
#         # ParameterValidator.assert_in_valid_list(f"{reference_sequences['format']}SequenceImport", valid_formats, location,
#         #                                         "format in reference_sequences")
#
#         assert os.path.isfile(reference_sequences["path"]), f"{location}: the file {reference_sequences['path']} does not exist. " \
#                                                             f"Specify the correct path under reference_sequences."
#
#         format_str = reference_sequences["format"]
#
#         import_class = ReflectionHandler.get_class_by_name("{}Import".format(format_str))
#         params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path + "datasets/",
#                                           DefaultParamsLoader._convert_to_snake_case(format_str))
#         params["paired"] = False
#         params["is_repertoire"] = False
#         processed_params = DatasetImportParams.build_object(**params)
#
#         sequences = ImportHelper.import_items(import_class, reference_sequences["path"], processed_params)
#
#
#
#
#
# in receptor class:
#
# ParameterValidator.assert_keys(list(reference_receptors.keys()), ["format", "path", "params"], location, "reference_receptors", exclusive=False)
#
#         # valid_formats = ReflectionHandler.discover_classes_by_partial_name("SequenceImport", "sequence_import/")
#         # ParameterValidator.assert_in_valid_list(f"{reference_receptors['format']}SequenceImport", valid_formats, location, "format in reference_receptors")
#
#         assert os.path.isfile(reference_receptors["path"]), f"{location}: the file {reference_receptors['path']} does not exist. " \
#                                                             f"Specify the correct path under reference_receptors."
#
#         # seq_import_params = reference_receptors["params"] if "params" in reference_receptors else {}
#         # if "paired" in seq_import_params:
#         #     assert seq_import_params["paired"] is True, f"{location}: paired must be True for SequenceImport"
#         # else:
#         #     seq_import_params["paired"] = True
#         #
#         # receptors = ReflectionHandler.get_class_by_name("{}SequenceImport".format(reference_receptors["format"]))\
#         #     .import_items(reference_receptors["path"], **seq_import_params)
#
#
#         format_str = reference_receptors["format"]
#
#
#