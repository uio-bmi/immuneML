import os

from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.IO.dataset_import.IRISSequenceImport import IRISSequenceImport
from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.ImportHelper import ImportHelper
from source.util.ParameterValidator import ParameterValidator
from source.util.ReflectionHandler import ReflectionHandler


class MatchedReferenceUtil:
    """
    Utility class for MatchedSequencesEncoder and MatchedReceptorsEncoder
    """

    @staticmethod
    def prepare_reference(reference_params: dict, location: str, paired: bool):
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

        if format_str == "IRIS": # todo refactor this when refactoring IRISSequenceImport
            receptors = IRISSequenceImport.import_items(**seq_import_params)
        else:
            import_class = ReflectionHandler.get_class_by_name("{}Import".format(format_str))
            params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "datasets",
                                              DefaultParamsLoader.convert_to_snake_case(format_str))
            for key, value in seq_import_params.items():
                params[key] = value
            params["paired"] = paired

            processed_params = DatasetImportParams.build_object(**params)

            receptors = ImportHelper.import_items(import_class, reference_params["params"]["path"], processed_params)

        return receptors
