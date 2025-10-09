import logging
import os
from pathlib import Path

from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.ImportHelper import ImportHelper
from immuneML.util.Logger import print_log
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReflectionHandler import ReflectionHandler


class MatchedReferenceUtil:
    """
    Utility class for MatchedSequencesEncoder and MatchedReceptorsEncoder
    """

    @staticmethod
    def prepare_reference(reference_params: dict, location: str, paired: bool):
        try:
            ParameterValidator.assert_keys(list(reference_params.keys()), ["format", "params"], location, "reference")

            seq_import_params = reference_params["params"] if "params" in reference_params else {}

            assert os.path.isfile(
                seq_import_params["path"]), f"{location}: the file {seq_import_params['path']} does not exist. " \
                                            f"Specify the correct path under reference."

            if "is_repertoire" in seq_import_params:
                assert seq_import_params[
                           "is_repertoire"] is False, f"{location}: is_repertoire must be False for SequenceImport"
            else:
                seq_import_params["is_repertoire"] = False

            if "paired" in seq_import_params:
                assert seq_import_params["paired"] == paired, f"{location}: paired must be {paired} for SequenceImport"
            else:
                seq_import_params["paired"] = paired

            format_str = reference_params["format"]

            import_class = ReflectionHandler.get_class_by_name(f"{format_str}Import")
            assert import_class is not None, (f"{MatchedReferenceUtil.__name__}: {format_str} could not be imported. "
                                              f"Check if the format name has been written correctly.")
            default_params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "datasets",
                                                      DefaultParamsLoader.convert_to_snake_case(format_str))

            params = {**default_params, **seq_import_params}

            path = Path(reference_params['params']['path'])
            params['result_path'] = PathBuilder.build(
                path.parent / 'iml_imported' if path.is_file() else path / 'iml_imported')

            if format_str == "SingleLineReceptor":
                receptors = list(import_class(params, 'tmp_receptor_dataset').import_dataset().get_data())
            else:
                receptors = list(import_class(params=params, dataset_name="tmp_dataset").import_dataset().get_data())

            assert len(
                receptors) > 0, f"MatchedReferenceUtil: The total number of imported reference {'receptors' if paired else 'sequences'} is 0, please ensure that reference import is specified correctly."

            check_imported_references(paired, receptors, seq_import_params)

            logging.info(
                f"MatchedReferenceUtil: successfully imported {len(receptors)} reference {'receptors' if paired else 'sequences'}.")

            return receptors
        except Exception as e:
            print_log(f"MatchedReferenceUtil: Error while preparing reference: {e}", logging.ERROR)
            raise e


def check_imported_references(paired, receptors, seq_import_params):
    check_for_duplicates(paired, receptors, seq_import_params)
    check_genes(paired, receptors, seq_import_params)


def check_genes(paired, receptors, seq_import_params):
    import re

    pattern = re.compile(r'^[A-Za-z]+[0-9]+(?:-[0-9]+)?(?:\*[0-9]+)?(?:/[A-Za-z]+[0-9]+)?$')

    if not paired:
        all_v_genes = set(seq.v_call for seq in receptors if seq.v_call is not None)
        all_j_genes = set(seq.j_call for seq in receptors if seq.j_call is not None)
    else:
        all_v_genes = (set(receptor.chain_1.v_call for receptor in receptors if receptor.chain_1.v_call is not None)
                       .union(set(receptor.chain_2.v_call for receptor in receptors if receptor.chain_2.v_call
                                  is not None)))
        all_j_genes = set(receptor.chain_1.j_call for receptor in receptors if receptor.chain_1.j_call is not None).union(
            set(receptor.chain_2.j_call for receptor in receptors if receptor.chain_2.j_call is not None))

    for gene_name, gene_list in [('V', all_v_genes), ('J', all_j_genes)]:
        if len(gene_list) > 0:
            assert all(pattern.match(gene) for gene in gene_list), \
                (f"{MatchedReferenceUtil.__name__}: The {gene_name} gene names in the reference sequences "
                 f"({seq_import_params['path']}) do not follow the IMGT nomenclature. Please ensure that the "
                 f"{gene_name} gene names are in the correct format (e.g., TRBV5-1*01 for V genes, TRBJ2-7*01 "
                 f"for J genes). Found {gene_name} genes: {gene_list}")


def check_for_duplicates(paired, receptors, seq_import_params):
    if not paired:
        all_sequences = [f'{seq.v_call}_{seq.get_sequence()}_{seq.j_call}' for seq in receptors]
        unique_sequences = set(all_sequences)
        if len(unique_sequences) < len(receptors):
            logging.warning(f"MatchedReferenceUtil: The reference sequences ({seq_import_params['path']}) "
                            f"contain duplicates: {len(all_sequences) - len(unique_sequences)} sequences are "
                            f"not unique. This will result in duplicate features.")
