import logging
import shutil
import warnings
from multiprocessing.pool import Pool
from pathlib import Path
from uuid import uuid4
from typing import List

import pandas as pd

from immuneML.data_model.bnp_util import read_yaml

from immuneML.data_model.bnp_util import write_yaml, bnp_write_to_file
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset, SequenceDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceSet import Repertoire, build_dynamic_airr_sequence_set_dataclass
from immuneML.environment.SequenceType import SequenceType
from immuneML.preprocessing.filters.CountAggregationFunction import CountAggregationFunction
from immuneML.preprocessing.filters.Filter import Filter
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from scripts.specification_util import update_docs_per_mapping


class DuplicateSequenceFilter(Filter):
    """
    Collapsing together duplicate sequences.

    Such duplicates may occur due to having identical amino acid sequences (but different nucleotide sequences),
    or because clones have not yet been called in the case of single-cell sequencing data.
    Parameter groupby_fields can be used to determine how to collapse duplicate sequences
    (e.g., amino acid or nucleotide level, including or excluding V/J genes, including or excluding sequence or receptor labels).

    Duplicate Receptors in ReceptorDatasets are only collapsed if both chains are duplicate.
    If you want to filter on the single chain level, please import your dataset as a SequenceDataset (paired=False).

    The duplicate_counts columns are merged together according to the count_agg function (e.g., sum).
    Other count values (umi_counts, consensus_counts), if present, are removed to avoid confusion.

    **Specification arguments:**

    - groupby_fields (list): All the fields across which duplicates are aggregated. All unique rows are kept, considering only their values in groupby_fieds columns. By default, groupby_fields is: [cdr3_aa, junction_aa, sequence_aa, locus, v_call, j_call].
      This means duplicates are filtered on the amino acid level (cdr3_aa, junction_aa, sequence_aa), and V and J gene information is considered (v_call, j_call).
      All not in this list are explicitly ignored during aggregation (e.g., cell_id, sequence_id, and many other fields). For these fields, only the 'first' row value is kept.

      - To ignore V and J gene information, remove [v_call, j_call] and possible other imported V/J related fields

      - To consider nucleotide sequence information, add [cdr3, junction, sequence] and possible other nucleotide information imported fields

      - For Sequence- and ReceptorDatasets, it is recommended to add the names of any label for which separate entries should be kept

    - count_agg (:py:obj:`~immuneML.preprocessing.filters.CountAggregationFunction.CountAggregationFunction`): determines how the sequence counts of duplicate sequences are aggregated. Valid options are defined by the CountAggregationFunction enum. By default, count_agg is SUM.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        preprocessing_sequences:
            my_preprocessing:
                - my_filter:
                    DuplicateSequenceFilter:
                        # required parameters:
                        filter_sequence_type: AMINO_ACID
                        # optional parameters (if not specified the values bellow will be used):
                        count_agg: SUM
                        region_type: IMGT_CDR3

    """

    @classmethod
    def build_object(cls, **kwargs):
        location = cls.__name__
        if "batch_size" in kwargs:
            warnings.warn( f"Parameter 'batch_size' for {location} is deprecated and will be ignored. To adjust parallelization, "
                           f"use parameter 'number_of_processes' in the instruction specification.")
            kwargs.pop("batch_size")
        if "filter_sequence_type" in kwargs:
            warnings.warn(f"Parameter 'filter_sequence_type' for {location} is deprecated and will be ignored. "
                          f"To adjust filtering, please use parameter 'groupby_fields' instead."
                          f"By default, sequences are filtered on 'amino acid' level (nucleotide sequences are ignored)")
            kwargs.pop("filter_sequence_type")

        # todo completely change parameters again, add back filter_sequence_type, region_type.... param for v_j?


        ParameterValidator.assert_keys(kwargs.keys(), ["groupby_fields", "count_agg"], location, "DuplicateSequenceFilter")
        ParameterValidator.assert_type_and_value(kwargs["count_agg"], str, location, "count_agg")
        ParameterValidator.assert_in_valid_list(kwargs["count_agg"].upper(), [item.name for item in CountAggregationFunction], location, "count_agg")

        ParameterValidator.assert_type_and_value(kwargs["groupby_fields"], list, location, "groupby_fields")
        ParameterValidator.assert_all_type_and_value(kwargs["groupby_fields"], str, location, "groupby_fields")

        if all([seq_name not in kwargs["groupby_fields"] for seq_name in ["cdr3_aa", "cdr3", "junction_aa", "junction", "sequence_aa", "sequence"]]):
            warnings.warn(f"{location}: groupby_fields did not contain any of the expected sequence fields (cdr3_aa, cdr3, junction_aa, junction, sequence_aa, sequence), "
                          f"this could mean groupby_fields was not correctly specified. "
                          f"\nUsing the following: {kwargs['groupby_fields']}"
                          f"\nTo remove this warning, add at least one of cdr3_aa, cdr3, junction_aa, junction, sequence_aa or sequence to groupby_fields. ")

        if any([id_name not in kwargs["groupby_fields"] for id_name in ["sequence_id", "cell_id"]]):
            warnings.warn(f"{location}: groupby_fields contained unexpected identifier field (sequence_id and/or cell_id). "
                          f"It is not recommended to include these fields, as the purpose of DuplicateSequenceFilter is to collapse together identical sequences (across identifiers). "
                          f"\nUsing the following: {kwargs['groupby_fields']}\n"
                          f"\nTo remove this warning, remove sequence_id and cell_id from groupby_fields. ")

        if "locus" not in kwargs["groupby_fields"]:
            warnings.warn(f"{location}: groupby_fields did not contain any of the expected field locus, "
                          f"this could mean groupby_fields was not correctly specified. "
                          f"\nUsing the following: {kwargs['groupby_fields']}\n"
                          f"\nTo remove this warning, add locus to groupby_fields. ")

        return DuplicateSequenceFilter(groupby_fields=kwargs["groupby_fields"], count_agg=CountAggregationFunction[kwargs["count_agg"].upper()])

    def __init__(self, groupby_fields: List[str], count_agg: CountAggregationFunction,
                 result_path: Path = None):
        super().__init__(result_path)
        self.groupby_fields = groupby_fields
        self.count_agg = count_agg

        # self.sequence_of_interest = bnp_util.get_sequence_field_name(self.region_type, self.filter_sequence_type)
        # self.sequence_to_ignore = bnp_util.get_sequence_field_name(self.region_type, [t for t in SequenceType if t != self.filter_sequence_type][0])

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path,
                        number_of_processes=1) -> Dataset:
        self.result_path = result_path if result_path is not None else self.result_path
        PathBuilder.build(self.result_path)

        # self.check_dataset_type(dataset, [RepertoireDataset], "DuplicateSequenceFilter")

        if type(dataset) == RepertoireDataset:
            return self.process_repertoire_dataset(dataset, number_of_processes)
        elif type(dataset) == ReceptorDataset:
            return self.process_receptor_dataset(dataset, number_of_processes)
        elif type(dataset) == SequenceDataset:
            return self.process_sequence_dataset(dataset, number_of_processes)

    def make_new_element_dataset(self, orig_dataset, filtered_df):
        name = f"{orig_dataset.name}_filtered"

        bnp_dc = orig_dataset.bnp_dataclass
        final_data_dict = filtered_df.to_dict(orient='list')
        data = bnp_dc(**final_data_dict)
        bnp_write_to_file(self.result_path / f"{name}.tsv", data)

        shutil.copyfile(orig_dataset.dataset_file, self.result_path / f'{name}.yaml')

        data_cls = orig_dataset.__class__

        return data_cls.build(filename=self.result_path / f"{name}.tsv",
                                     metadata_filename=self.result_path / f'{name}.yaml',
                                     name=name, bnp_dc=bnp_dc, labels=orig_dataset.labels)

    def process_sequence_dataset(self, dataset: SequenceDataset, number_of_processes=1):
        metadata = read_yaml(dataset.dataset_file)
        custom_labels = list(metadata["type_dict_dynamic_fields"].keys()) # todo deal with custom labels? checking/warning?

        filtered_df = self._process_sequence_df(data=dataset.data.topandas())

        return self.make_new_element_dataset(dataset, filtered_df)

    def process_receptor_dataset(self, dataset: ReceptorDataset, number_of_processes=1):
        metadata = read_yaml(dataset.dataset_file)
        custom_labels = list(metadata["type_dict_dynamic_fields"].keys())

        filtered_df = self._process_receptor_df(data=dataset.data.topandas())

        return self.make_new_element_dataset(dataset, filtered_df)

    def process_repertoire_dataset(self, dataset: RepertoireDataset, number_of_processes=1) -> RepertoireDataset:
        processed_dataset = dataset.clone()

        with Pool(number_of_processes) as pool:
            repertoires = pool.map(self._process_repertoire, dataset.repertoires)

        processed_dataset.repertoires = repertoires

        return processed_dataset

    def _prepare_groupby_field(self, columns, groupby_fields):
        for field in groupby_fields:
            if field not in columns:
                warnings.warn(f"DuplicateSequenceFilter: groupby field {field} was specified but not found in data (columns: {list(columns)}). Omitting this field...")

        return [field for field in groupby_fields if field in columns]

    def _prepare_agg_dict(self, columns, groupby_fields, loci_suffixes=None):
        agg_dict = {}

        for field in columns:
            if field not in groupby_fields:
                agg_dict[field] = "first"

        if loci_suffixes is not None:
            for suffix in loci_suffixes:
                if "duplicate_count" + suffix in columns:
                    agg_dict["duplicate_count" + suffix] = self.count_agg.value

        if "duplicate_count" in columns:
            agg_dict["duplicate_count"] = self.count_agg.value

        return agg_dict

    def _process_repertoire(self, repertoire: Repertoire) -> Repertoire:
        no_duplicates = self._process_sequence_df(data=repertoire.data.topandas())

        no_duplicates.to_csv(f"{self.result_path}/{repertoire.data_filename.stem}_filtered.tsv", sep='\t', index=False)
        write_yaml(Path(f"{self.result_path}/{repertoire.metadata_filename.stem}_filtered.yaml"), repertoire.metadata)

        return Repertoire(data_filename=Path(f"{self.result_path}/{repertoire.data_filename.stem}_filtered.tsv"),
                          metadata_filename=Path(f"{self.result_path}/{repertoire.metadata_filename.stem}_filtered.yaml"),
                          identifier=str(uuid4().hex),
                          dynamic_fields=repertoire.dynamic_fields)

    def _process_df_counts(self, data, colname_suffix=""):
        # presuming default duplicate count = 1 (better than presuming NA)
        # This is important for single-cell datasets where only umi_count is set, and each entry is 1 duplicate
        if set(data[f"duplicate_count{colname_suffix}"]) == {-1}:
            logging.warning(
                "DuplicateSequenceFilter: duplicate_count was found to be -1 for all sequences (due to not being imported). "
                "Setting all -1 duplicate counts to 1 before duplicate sequence aggregation...")
            data[f"duplicate_count{colname_suffix}"] = data[f"duplicate_count{colname_suffix}"].replace(-1, 1)

        # set other count values explicitly to -1 as these counts are either wrong or nonsensical after aggregation
        if f"umi_count{colname_suffix}" in data and set(data[f"umi_count{colname_suffix}"]) != {-1}:
            logging.warning(
                "DuplicateSequenceFilter: umi_count was set to -1 as these counts are either wrong or nonsensical after aggregation")
            data[f"umi_count{colname_suffix}"] = -1

        if f"consensus_count{colname_suffix}" in data and set(data[f"consensus_count{colname_suffix}"]) != {-1}:
            logging.warning(
                "DuplicateSequenceFilter: consensus_count was set to -1 as these counts are either wrong or nonsensical after aggregation")
            data[f"consensus_count{colname_suffix}"] = -1

    def _process_sequence_df(self, data):
        self._process_df_counts(data)

        groupby_fields = self._prepare_groupby_field(data.columns, self.groupby_fields)
        agg_dict = self._prepare_agg_dict(data.columns, groupby_fields)

        logging.info(f"DuplicateSequenceFilter: Using the following groupby fields: {groupby_fields}")
        logging.info(f"DuplicateSequenceFilter: Using the following aggregration dict: {agg_dict}")

        no_duplicates = data.groupby(groupby_fields, sort=False).agg(agg_dict).reset_index()

        return no_duplicates

    def _process_receptor_df(self, data):
        loci = sorted(set(data["locus"]))
        assert len(loci) == 2, f"DuplicateSequenceFilter: Expected 2 loci, found the following: {loci}"

        data = data.pivot(index='cell_id', columns='locus')
        data.columns = [f"{col[0]}#{col[1]}" for col in data.columns]
        data.reset_index(inplace=True)
        data[f"locus#{loci[0]}"] = loci[0]
        data[f"locus#{loci[1]}"] = loci[1]

        self._process_df_counts(data, "#" + loci[0])
        self._process_df_counts(data, "#" + loci[1])

        groupby_fields = self._prepare_groupby_field(data.columns,
                                                     [f"{field}#{loci[0]}" for field in self.groupby_fields] +
                                                     [f"{field}#{loci[1]}" for field in self.groupby_fields])
        agg_dict = self._prepare_agg_dict(data.columns, groupby_fields, loci_suffixes=["#" + locus for locus in loci])

        logging.info(f"DuplicateSequenceFilter: Using the following groupby fields: {groupby_fields}")
        logging.info(f"DuplicateSequenceFilter: Using the following aggregration dict: {agg_dict}")

        no_duplicates = data.groupby(groupby_fields, sort=False).agg(agg_dict).reset_index()

        locus0_df = no_duplicates[['cell_id'] + [col for col in no_duplicates.columns if col.endswith(loci[0])]].copy().reset_index(drop=True)
        locus1_df = no_duplicates[['cell_id'] + [col for col in no_duplicates.columns if col.endswith(loci[1])]].copy().reset_index(drop=True)

        locus0_df.columns = [col.replace(f"#{loci[0]}", "") for col in locus0_df.columns]
        locus1_df.columns = [col.replace(f"#{loci[1]}", "") for col in locus1_df.columns]

        assert all(locus0_df.columns == locus1_df.columns)

        combined_loci = pd.concat([locus0_df, locus1_df], axis=0, ignore_index=True)
        combined_loci.sort_values(by=['cell_id', 'locus'], ascending=True, inplace=True)

        return combined_loci


    @staticmethod
    def get_documentation():
        doc = str(DuplicateSequenceFilter.__doc__)

        mapping = {
            "Valid options are defined by the CountAggregationFunction enum.": f"Valid values are: {[e.name.lower() for e in CountAggregationFunction]}.",
        }

        doc = update_docs_per_mapping(doc, mapping)

        return doc
