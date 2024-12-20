import copy
import glob
import shutil
import subprocess
from pathlib import Path
from typing import List

import pandas as pd

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.SequenceSet import ReceptorSequence, Repertoire
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.encodings.reference_encoding.MatchedReferenceUtil import MatchedReferenceUtil
from immuneML.preprocessing.Preprocessor import Preprocessor
from immuneML.util.CompAIRRHelper import CompAIRRHelper
from immuneML.util.CompAIRRParams import CompAIRRParams
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class ReferenceSequenceAnnotator(Preprocessor):
    """
    Annotates each sequence in each repertoire if it matches any of the reference sequences provided as input parameter. This report uses CompAIRR internally. To match CDR3 sequences (and not JUNCTION), CompAIRR v1.10 or later is needed.

    **Specification arguments:**

    - reference_sequences (dict): A dictionary describing the reference dataset file. Import should be specified the same way as regular dataset import. It is only allowed to import a receptor dataset here (i.e., is_repertoire is False and paired is True by default, and these are not allowed to be changed).

    - max_edit_distance (int): The maximum edit distance between a target sequence (from the repertoire) and the reference sequence.

    - compairr_path (str): optional path to the CompAIRR executable. If not given, it is assumed that CompAIRR has been installed such that it can be called directly on the command line with the command 'compairr', or that it is located at /usr/local/bin/compairr.

    - threads (int): how many threads to be used by CompAIRR for sequence matching

    - ignore_genes (bool): Whether to ignore V and J gene information. If False, the V and J genes between two receptor chains have to match. If True, gene information is ignored. By default, ignore_genes is False.

    - output_column_name (str): in case there are multiple annotations, it is possible here to define the name of the column in the output repertoire files for this specific annotation

    - repertoire_batch_size (int): how many repertoires to process simultaneously; depending on the repertoire size, this parameter might be use to limit the memory usage

    - region_type (str): which region type to check, default is IMGT_CDR3


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        preprocessing_sequences:
            my_preprocessing:
                - step1:
                    ReferenceSequenceAnnotator:
                        reference_sequences:
                            format: VDJDB
                            params:
                                path: path/to/file.csv
                        compairr_path: optional/path/to/compairr
                        ignore_genes: False
                        max_edit_distance: 0
                        output_column_name: matched
                        threads: 4
                        repertoire_batch_size: 5
                        region_type: IMGT_CDR3

    """

    def __init__(self, reference_sequences: List[ReceptorSequence], max_edit_distance: int, compairr_path: str, ignore_genes: bool, threads: int,
                 output_column_name: str, repertoire_batch_size: int, region_type: RegionType):
        super().__init__()
        self._reference_sequences = reference_sequences
        self._max_edit_distance = max_edit_distance
        self._output_column_name = output_column_name
        self._repertoire_batch_size = repertoire_batch_size
        self._region_type = region_type
        self._compairr_params = CompAIRRParams(compairr_path=CompAIRRHelper.determine_compairr_path(compairr_path), keep_compairr_input=True,
                                               differences=max_edit_distance, indels=False, ignore_counts=True, ignore_genes=ignore_genes,
                                               threads=threads, output_filename="compairr_out.tsv", log_filename="compairr_log.txt",
                                               output_pairs=False, do_repertoire_overlap=False, do_sequence_matching=True, pairs_filename="pairs.tsv")

    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_keys(list(kwargs.keys()),
                                       ['reference_sequences', 'max_edit_distance', 'compairr_path', 'ignore_genes', 'output_column_name', 'threads',
                                        'repertoire_batch_size', 'region_type'],
                                       ReferenceSequenceAnnotator.__name__, ReferenceSequenceAnnotator.__name__)
        ParameterValidator.assert_in_valid_list(kwargs['region_type'].upper(), [el.name for el in RegionType],
                                                ReferenceSequenceAnnotator.__name__, 'region_type')
        ref_seqs = MatchedReferenceUtil.prepare_reference(reference_params=kwargs['reference_sequences'],
                                                          location=ReferenceSequenceAnnotator.__name__, paired=False)
        return ReferenceSequenceAnnotator(**{**kwargs, 'reference_sequences': ref_seqs,
                                             'region_type': RegionType[kwargs['region_type'].upper()]})

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path, number_of_processes=1) -> RepertoireDataset:
        self._compairr_params.is_cdr3 = self._region_type == RegionType.IMGT_CDR3
        sequence_filepath = self._prepare_sequences_for_compairr(result_path / 'tmp')
        repertoires_filepaths = self._prepare_repertoires_for_compairr(dataset, result_path / 'tmp')

        compairr_output_files = self._annotate_repertoires(sequence_filepath, repertoires_filepaths, result_path)

        processed_dataset = self._make_annotated_dataset(dataset, result_path, compairr_output_files)

        shutil.rmtree(result_path / 'tmp')

        return processed_dataset

    def _annotate_repertoires(self, sequences_filepath, repertoire_filepaths: List[Path], result_path: Path):
        tmp_path = PathBuilder.build(result_path / 'tmp')
        updated_output_files = []

        for index, rep_file in enumerate(repertoire_filepaths):
            batch_tmp_path = PathBuilder.build(tmp_path / str(index))
            args = CompAIRRHelper.get_cmd_args(self._compairr_params, [rep_file, sequences_filepath], batch_tmp_path)
            compairr_result = subprocess.run(args, capture_output=True, text=True)
            output_file = CompAIRRHelper.verify_compairr_output_path(compairr_result, self._compairr_params, batch_tmp_path)
            updated_output_file = result_path / f'updated_compairr_output_{index}.tsv'

            with open(output_file, 'r') as file:
                output_lines = file.readlines()

            with updated_output_file.open(mode='w') as file:
                output_lines[0] = output_lines[0].replace("#", '')
                file.writelines(output_lines)

            updated_output_files.append(updated_output_file)

        return updated_output_files

    def _make_annotated_dataset(self, dataset: RepertoireDataset, result_path: Path, compairr_output_files: List[Path]) -> RepertoireDataset:
        repertoires = []
        repertoire_path = PathBuilder.build(result_path / 'repertoires')

        for index, repertoire in enumerate(dataset.repertoires):

            compairr_out_df = pd.read_csv(compairr_output_files[index], sep='\t', comment="#")

            data = repertoire.data.add_fields({self._output_column_name: compairr_out_df.iloc[:, 1].values.astype(int)},
                                              {self._output_column_name: int})
            type_dict = type(data).get_field_type_dict(all_fields=False)

            repertoires.append(Repertoire.build_from_dc_object(repertoire_path, metadata=repertoire.metadata,
                                                               filename_base=repertoire.data_filename.name, data=data,
                                                               type_dict=type_dict))

        return RepertoireDataset.build_from_objects(**{'repertoires': repertoires, 'path': result_path})

    def _prepare_sequences_for_compairr(self, result_path: Path) -> Path:
        path = PathBuilder.build(result_path) / 'reference_sequences.tsv'

        # TODO: remove this when import is refactored, this now ensures that string matching is done on sequences as imported
        reference_sequences = []
        for seq in self._reference_sequences:
            tmp_seq = copy.deepcopy(seq)
            tmp_seq.duplicate_count = seq.duplicate_count if not self._compairr_params.ignore_counts else 1
            reference_sequences.append(tmp_seq)

        SequenceDataset.build_from_objects(reference_sequences, PathBuilder.build(result_path),
                                           region_type=self._region_type)

        result_files = glob.glob(str(result_path / "*.tsv"))
        assert len(result_files) == 1, \
            f"Error occurred while exporting sequences for matching using CompAIRR. Resulting files: {result_files}"
        shutil.move(result_files[0], path)

        return path

    def _prepare_repertoires_for_compairr(self, dataset: RepertoireDataset, result_path: Path) -> List[Path]:
        PathBuilder.build(result_path)
        paths = []
        for i, repertoire in enumerate(dataset.repertoires):
            path = result_path / f'repertoires_{i}.tsv'
            CompAIRRHelper.write_repertoire_file(repertoires=[repertoire], filename=path,
                                                 compairr_params=self._compairr_params, export_sequence_id=True)
            paths.append(path)
        return paths

    def _check_column_name(self, dataset):
        for repertoire in dataset.repertoires:
            assert repertoire.get_attribute(self._output_column_name) is None, \
                (f"{ReferenceSequenceAnnotator.__name__}: attribute {self._output_column_name} already exists in "
                 f"repertoire ({repertoire.identifier}); choose another name.")
