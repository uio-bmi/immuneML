import logging
import os
from pathlib import Path

from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.ImportHelper import ImportHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReflectionHandler import ReflectionHandler


class MatchedReferenceUtil:
    """
    Utility class for MatchedSequencesEncoder and MatchedReceptorsEncoder
    """

    @staticmethod
    def prepare_reference(reference_params: dict, location: str, paired: bool):
        ParameterValidator.assert_keys(list(reference_params.keys()), ["format", "params"], location, "reference")

        seq_import_params = reference_params["params"] if "params" in reference_params else {}

        assert os.path.isfile(seq_import_params["path"]), f"{location}: the file {seq_import_params['path']} does not exist. " \
                                                          f"Specify the correct path under reference."

        if "is_repertoire" in seq_import_params:
            assert seq_import_params["is_repertoire"] is False, f"{location}: is_repertoire must be False for SequenceImport"
        else:
            seq_import_params["is_repertoire"] = False

        if "paired" in seq_import_params:
            assert seq_import_params["paired"] == paired, f"{location}: paired must be {paired} for SequenceImport"
        else:
            seq_import_params["paired"] = paired

        format_str = reference_params["format"]

        import_class = ReflectionHandler.get_class_by_name(f"{format_str}Import")
        default_params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "datasets",
                                                  DefaultParamsLoader.convert_to_snake_case(format_str))

        params = {**default_params, **seq_import_params}

        processed_params = DatasetImportParams.build_object(**params)
        path = Path(reference_params['params']['path'])
        processed_params.result_path = PathBuilder.build(path.parent / 'iml_imported' if path.is_file() else path / 'iml_imported')

        if format_str == "SingleLineReceptor":
            receptors = list(import_class.import_dataset(processed_params, 'tmp_receptor_dataset').get_data())
        else:
            receptors = ImportHelper.import_items(import_class, reference_params["params"]["path"], processed_params)

        assert len(receptors) > 0, f"MatchedReferenceUtil: The total number of imported reference {'receptors' if paired else 'sequences'} is 0, please ensure that reference import is specified correctly."
        logging.info(f"MatchedReferenceUtil: successfully imported {len(receptors)} reference {'receptors' if paired else 'sequences'}.")

        return receptors
