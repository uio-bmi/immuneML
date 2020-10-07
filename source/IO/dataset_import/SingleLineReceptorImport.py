import glob
import logging
import os
from typing import List

import pandas as pd

from scripts.specification_util import update_docs_per_mapping
from source.IO.dataset_export.PickleExporter import PickleExporter
from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.IO.dataset_import.ReceptorDatasetImportParams import ReceptorDatasetImportParams
from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.receptor.ChainPair import ChainPair
from source.data_model.receptor.ReceptorBuilder import ReceptorBuilder
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.util.PathBuilder import PathBuilder


class SingleLineReceptorImport(DataImport):
    """
    Imports receptor dataset from a file or a set of files (located in the same directory). For column mapping, it has no default params, so it
    has to be specified manually which columns in the files correspond to which chain, gene, identifier, epitope. All valid immuneML values are given
    in the specification example below (mandatory fields are: `alpha_amino_acid_sequence`, `beta_amino_acid_sequence`, `alpha_nucleotide_sequence`,
    `beta_nucleotide_sequence`, `alpha_v_gene`, `alpha_j_gene`, `beta_v_gene`, `beta_j_gene`, `identifier`). Fields which are not listed here will be
    stored as metadata in the created receptor objects.
    Chain names are given in :py:obj:`~source.data_model.receptor.receptor_sequence.Chain.Chain`.
    Chain pairs are given in :py:obj:`~source.data_model.receptor.ChainPair.ChainPair`.

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_receptor_dataset:
            format: GenericReceptor
            params:
                path: path_to_csv_file.csv # path to a file with receptor data or a path to the directory with multiple receptor data files
                result_path: resulting_dataset/ # where to store the imported dataset
                separator: ',' # separator in the original receptor data files
                columns_to_load: [subject,epitope,count,v_a_gene,j_a_gene,cdr3_a_aa,v_b_gene,j_b_gene,cdr3_b_aa,clone_id] # which columns to load from the original receptor data file
                column_mapping: # how to rename the columns so that they can be recognized by immuneML in format: original name in receptor file: immuneML name
                    cdr3_a_aa: alpha_amino_acid_sequence # the sequence in the input receptor file and corresponding chain name in immuneML
                    cdr3_b_aa: beta_amino_acid_sequence
                    cdr3_a_nucseq: alpha_nucleotide_sequence
                    cdr3_b_nucseq: beta_nucleotide_sequence
                    v_a_gene: alpha_v_gene # for genes, chain name is the prefix when importing paired data
                    v_b_gene: beta_v_gene
                    j_a_gene: alpha_j_gene
                    j_b_gene: beta_j_gene
                    clone_id: identifier
                    epitope: epitope # everything other than sequences, V and J gene per chain, and an identifier will be stored in the receptor's metadata
                chains: ALPHA_BETA # which receptor chains are in the input receptor data file(s)
                region_type: CDR3
                sequence_file_size: 50000
                organism: mouse # mouse or human

    """

    @staticmethod
    def import_dataset(params, dataset_name: str) -> ReceptorDataset:
        generic_params = ReceptorDatasetImportParams.build_object(**params)
        filenames = SingleLineReceptorImport._extract_filenames(generic_params)

        PathBuilder.build(generic_params.result_path, warn_if_exists=True)

        dataset = SingleLineReceptorImport._import_from_files(filenames, generic_params)
        dataset.name = dataset_name
        dataset.params = {"organism": generic_params.organism}

        PickleExporter.export(dataset, generic_params.result_path)

        return dataset

    @staticmethod
    def _import_from_files(filenames: List[str], generic_params: ReceptorDatasetImportParams) -> ReceptorDataset:
        elements = []

        for file in filenames:
            df = pd.read_csv(file, sep=generic_params.separator, usecols=generic_params.columns_to_load)
            df.dropna()
            df.drop_duplicates()
            df.rename(columns=generic_params.column_mapping, inplace=True)
            for index, row in df.iterrows():
                chain_vals = [ch for ch in generic_params.chains.value]
                chain_names = [Chain.get_chain(ch).name.lower() for ch in generic_params.chains.value]
                sequences = {chain_vals[i]: ReceptorSequence(amino_acid_sequence=row[
                                     chain_name + "_amino_acid_sequence"] if chain_name + "_amino_acid_sequence" in row else None,
                                                  nucleotide_sequence=row[
                                                      chain_name + "_nucleotide_sequence"] if chain_name + "_nucleotide_sequence" in row else None,
                                                  metadata=SequenceMetadata(
                                                      v_gene=row[f"{chain_name}_v_gene"],
                                                      j_gene=row[f"{chain_name}_j_gene"],
                                                      chain=chain_name,
                                                      count=row["count"],
                                                      region_type=generic_params.region_type.value))
                             for i, chain_name in enumerate(chain_names)}

                elements.append(ReceptorBuilder.build_object(sequences, row["identifier"],
                                                             {key: row[key] for key in row.keys()
                                                              if all(item not in key for item in
                                                                     ["v_gene", 'j_gene', "count", "identifier"] + chain_names)}))

        return ReceptorDataset.build(elements, generic_params.sequence_file_size, generic_params.result_path)

    @staticmethod
    def _extract_filenames(params: DatasetImportParams) -> List[str]:
        if os.path.isdir(params.path):
            filenames = list(glob.glob(params.path + "*.tsv" if params.separator == "\t" else "*.csv"))
        elif os.path.isfile(params.path):
            filenames = [params.path]
        else:
            raise ValueError(f"SingleLineReceptorImport: path '{params.path}' given in YAML specification is not a valid path to receptor files. "
                             f"This parameter can either point to a single file with receptor data or to a directory where a list of receptor data "
                             f"files are stored directly.")

        logging.info(f"SingleLineReceptorImport: importing from receptor files: \n{str([os.path.basename(file) for file in filenames])[1:-1]}")
        return filenames

    @staticmethod
    def get_documentation():
        doc = str(SingleLineReceptorImport.__doc__)

        valid_chain_names = str([item.name for item in Chain])[1:-1].replace("'", "`")
        valid_chain_pair_names = str([item.name for item in ChainPair])[1:-1].replace("'", "`")

        mapping = {
            "Chain names are given in :py:obj:`~source.data_model.receptor.receptor_sequence.Chain.Chain`.":
                f"Valid chain names are: {valid_chain_names}.",
            "Chain pairs are given in :py:obj:`~source.data_model.receptor.ChainPair.ChainPair`.":
                f"Valid chain names are: {valid_chain_pair_names}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
