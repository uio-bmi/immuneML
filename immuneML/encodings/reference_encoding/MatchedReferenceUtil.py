import os

from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.ImportHelper import ImportHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


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

        if "is_repertoire" in seq_import_params:
            assert seq_import_params["is_repertoire"] == False, f"{location}: is_repertoire must be False for SequenceImport"
        else:
            seq_import_params["is_repertoire"] = False

        if "paired" in seq_import_params:
            assert seq_import_params["paired"] == paired, f"{location}: paired must be {paired} for SequenceImport"
        else:
            seq_import_params["paired"] = paired

        format_str = reference_params["format"]

        import_class = ReflectionHandler.get_class_by_name("{}Import".format(format_str))
        default_params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "datasets",
                                          DefaultParamsLoader.convert_to_snake_case(format_str))

        params = {**default_params, **seq_import_params}

        processed_params = DatasetImportParams.build_object(**params)

        receptors = ImportHelper.import_items(import_class, reference_params["params"]["path"], processed_params)

        return receptors
