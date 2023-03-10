import copy
import glob
import shutil
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.reference_encoding.MatchedReferenceUtil import MatchedReferenceUtil
from immuneML.preprocessing.Preprocessor import Preprocessor
from immuneML.util.CompAIRRHelper import CompAIRRHelper
from immuneML.util.CompAIRRParams import CompAIRRParams
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class ReferenceSequenceAnnotator(Preprocessor):
    """
    Annotates each sequence in each repertoire if it matches any of the reference sequences provided as input parameter.

    Arguments:

        reference (dict): A dictionary describing the reference dataset file. Import should be specified the same way as regular dataset import. It is only allowed to import a receptor dataset here (i.e., is_repertoire is False and paired is True by default, and these are not allowed to be changed).

        max_edit_distance (int): The maximum edit distance between a target sequence (from the repertoire) and the reference sequence.

        compairr_path (str): optional path to the CompAIRR executable. If not given, it is assumed that CompAIRR has been installed such that it can be called directly on the command line with the command 'compairr', or that it is located at /usr/local/bin/compairr.

        ignore_genes (bool): Whether to ignore V and J gene information. If False, the V and J genes between two receptor chains have to match. If True, gene information is ignored. By default, ignore_genes is False.

        output_column_name (str): in case there are multiple annotations, it is possible here to define the name of the column in the output repertoire files for this specific annotation

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        preprocessing_sequences:
            my_preprocessing:
                - step1:
                    ReferenceSequenceAnnotator:
                        reference:
                            format: VDJDB
                            params:
                                path: path/to/file.csv
                        compairr_path: optional/path/to/compairr
                        ignore_genes: False
                        max_edit_distance: 0
                        output_column_name: matched

    """

    def __init__(self, reference_sequences: List[ReceptorSequence], max_edit_distance: int, compairr_path: str, ignore_genes: bool, threads: int,
                 output_column_name: str):
        super().__init__()
        self._reference_sequences = reference_sequences
        self._max_edit_distance = max_edit_distance
        self._output_column_name = output_column_name
        self._compairr_params = CompAIRRParams(compairr_path=CompAIRRHelper.determine_compairr_path(compairr_path), keep_compairr_input=True,
                                               differences=max_edit_distance, indels=False, ignore_counts=True, ignore_genes=ignore_genes,
                                               threads=threads, output_filename="compairr_out.tsv", log_filename="compairr_log.txt",
                                               output_pairs=True, do_repertoire_overlap=False, do_sequence_matching=True, pairs_filename="pairs.tsv")

    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_keys(list(kwargs.keys()), ['reference', 'max_edit_distance', 'compairr_path', 'ignore_genes', 'output_column_name'],
                                       ReferenceSequenceAnnotator.__name__, ReferenceSequenceAnnotator.__name__)
        ref_seqs = MatchedReferenceUtil.prepare_reference(reference_params=kwargs['reference'], location=ReferenceSequenceAnnotator.__name__,
                                                          paired=False)
        return ReferenceSequenceAnnotator(**{**{k: v for k, v in kwargs.items() if k != 'reference'}, 'reference_sequences': ref_seqs})

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path, number_of_processes=1) -> RepertoireDataset:
        sequence_filepath = self._prepare_sequences_for_compairr(result_path / 'tmp')
        repertoires_filepath = self._prepare_repertoires_for_compairr(dataset, result_path / 'tmp')

        compairr_output_file = self._annotate_repertoires(sequence_filepath, repertoires_filepath, result_path / 'tmp')

        processed_dataset = self._make_annotated_dataset(dataset, result_path, compairr_output_file)

        shutil.rmtree(result_path / 'tmp')

        return processed_dataset

    def _annotate_repertoires(self, sequences_filepath, repertoire_filepath: Path, result_path: Path):
        args = CompAIRRHelper.get_cmd_args(self._compairr_params, [sequences_filepath, repertoire_filepath], result_path)
        compairr_result = subprocess.run(args, capture_output=True, text=True)
        output_file = CompAIRRHelper.verify_compairr_output_path(compairr_result, self._compairr_params, result_path)

        with open(output_file, 'r') as file:
            output_lines = file.readlines()

        with open(result_path / 'updated_compairr_output.tsv', 'w') as file:
            output_lines[0] = output_lines[0].replace("#", '')
            file.writelines(output_lines)

        return result_path / 'updated_compairr_output.tsv'

    def _make_annotated_dataset(self, dataset: RepertoireDataset, result_path: Path, compairr_output_file) -> RepertoireDataset:
        repertoires = []
        repertoire_path = PathBuilder.build(result_path / 'repertoires')
        compairr_out_df = pd.read_csv(compairr_output_file, sep='\t')

        for index, repertoire in enumerate(dataset.repertoires):
            if compairr_out_df[repertoire.identifier].any():
                sequence_selection = compairr_out_df[repertoire.identifier]
                matches = np.array([repertoire.get_sequence_aas() == seq.amino_acid_sequence for seq in
                                    np.array(self._reference_sequences)[sequence_selection.values]]).any(axis=0)
                sequences = self._add_params_to_sequence_objects(repertoire.sequences, matches)
            else:
                sequences = self._add_params_to_sequence_objects(repertoire.sequences, np.zeros(len(repertoire.sequences), dtype=bool))

            repertoires.append(Repertoire.build_from_sequence_objects(sequences, repertoire_path, repertoire.metadata))

        return RepertoireDataset.build_from_objects(**{'repertoires': repertoires, 'path': result_path})

    def _add_params_to_sequence_objects(self, sequence_objects: List[ReceptorSequence], matches_reference):
        sequences = copy.deepcopy(sequence_objects)
        for seq_index, seq in enumerate(sequences):
            seq.metadata.custom_params[self._output_column_name] = int(matches_reference[seq_index])
        return sequences

    def _prepare_sequences_for_compairr(self, result_path: Path) -> Path:
        PathBuilder.build(result_path)
        path = result_path / 'reference_sequences.tsv'
        AIRRExporter.export(SequenceDataset.build_from_objects(self._reference_sequences, len(self._reference_sequences),
                                                               PathBuilder.build(result_path / 'tmp_seq_dataset')), result_path)

        result_files = glob.glob(str(result_path / "*.tsv"))
        assert len(result_files) == 1, f"Error occurred while exporting sequences for matching using CompAIRR. Resulting files: {result_files}"
        shutil.move(result_files[0], path)

        return path

    def _prepare_repertoires_for_compairr(self, dataset: RepertoireDataset, result_path: Path) -> Path:
        PathBuilder.build(result_path)
        path = result_path / 'repertoires.tsv'
        CompAIRRHelper.write_repertoire_file(dataset, path, self._compairr_params)
        return path

    def _check_column_name(self, dataset):
        for repertoire in dataset.repertoires:
            assert repertoire.get_attribute(self._output_column_name) is None, \
                f"{ReferenceSequenceAnnotator.__name__}: attribute {self._output_column_name} already exists in repertoire ({repertoire.identifier}); choose another name."
