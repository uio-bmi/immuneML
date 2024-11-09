import copy
import shutil
import warnings
from multiprocessing.pool import Pool
from pathlib import Path
from uuid import uuid4

import bionumpy as bnp
import pandas as pd

from immuneML.data_model import bnp_util
from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.SequenceParams import RegionType
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
    Collapses duplicate nucleotide or amino acid sequences within each repertoire in the given RepertoireDataset.
    This filter can be applied to Repertoires and RepertoireDatasets.

    Sequences are considered duplicates if the following fields are identical:

    - amino acid or nucleotide sequence (whichever is specified)

    - v and j genes (note that the full field including subgroup + gene is used for matching, i.e. V1 and V1-1 are not considered duplicates)

    - chain

    - region type

    For all other fields (the non-specified sequence type, custom lists, sequence identifier) only the first occurring
    value is kept.

    Note that this means the count value of a sequence with a given sequence identifier might not be the same as before
    removing duplicates, unless count_agg = FIRST is used.

    **Specification arguments:**

    - filter_sequence_type (:py:obj:`~immuneML.environment.SequenceType.SequenceType`): Whether the sequences should be collapsed on the nucleotide or amino acid level. Valid options are defined by the SequenceType enum.

    - region_type (str): which part of the sequence to examine, by default, this is IMGT_CDR3

    - count_agg (:py:obj:`~immuneML.preprocessing.filters.CountAggregationFunction.CountAggregationFunction`): determines how the sequence counts of duplicate sequences are aggregated. Valid options are defined by the CountAggregationFunction enum.


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

        ParameterValidator.assert_keys(kwargs.keys(), ["filter_sequence_type", "count_agg", "region_type"], location,
                                       "DuplicateSequenceFilter")
        ParameterValidator.assert_in_valid_list(kwargs["filter_sequence_type"].upper(), [item.name for item in SequenceType],
                                                location, "filter_sequence_type")
        ParameterValidator.assert_in_valid_list(kwargs["region_type"].upper(), [item.name for item in RegionType],
                                                location, "region_type")
        ParameterValidator.assert_in_valid_list(kwargs["count_agg"].upper(), [item.name for item in CountAggregationFunction], location,
                                                "count_agg")
        return DuplicateSequenceFilter(filter_sequence_type=SequenceType[kwargs["filter_sequence_type"].upper()],
                                       region_type=RegionType[kwargs['region_type'].upper()],
                                       count_agg=CountAggregationFunction[kwargs["count_agg"].upper()])

    def __init__(self, filter_sequence_type: SequenceType, count_agg: CountAggregationFunction,
                 result_path: Path = None, region_type: RegionType = RegionType.IMGT_CDR3):
        super().__init__(result_path)
        self.filter_sequence_type = filter_sequence_type
        self.region_type = region_type
        self.count_agg = count_agg

        self.sequence_of_interest = bnp_util.get_sequence_field_name(self.region_type, self.filter_sequence_type)
        self.sequence_to_ignore = bnp_util.get_sequence_field_name(self.region_type, [t for t in SequenceType if t != self.filter_sequence_type][0])

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

    def process_receptor_dataset(self, dataset: ReceptorDataset, number_of_processes=1):
        pass

    def process_sequence_dataset(self, dataset: SequenceDataset, number_of_processes=1):
        no_duplicates = self._process_df(data=dataset.data.topandas(), custom_lists=[])

        name = f"{dataset.name}_filtered"
        new_filename = f"{self.result_path}/{name}.tsv"

        data = AIRRSequenceSet.from_data_frame(no_duplicates)
        bnp_write_to_file(new_filename, data)


        new_dataset_file = self.result_path / f'{name}.yaml'
        shutil.copyfile(dataset.dataset_file, new_dataset_file)

        # todo something here is not working
        return SequenceDataset.build(filename=new_filename, metadata_filename=new_dataset_file, name=name, labels=dataset.labels)

        # return SequenceDataset(filename=new_filename, name=name, labels=copy.deepcopy(dataset.labels),
        #                        dynamic_fields=dataset.dynamic_fields, dataset_file=new_dataset_file,
        #                        bnp_dataclass=dataset.bnp_dataclass)
        #

    def process_repertoire_dataset(self, dataset: RepertoireDataset, number_of_processes=1) -> RepertoireDataset:
        processed_dataset = dataset.clone()

        with Pool(number_of_processes) as pool:
            repertoires = pool.map(self._process_repertoire, dataset.repertoires)

        processed_dataset.repertoires = repertoires

        return processed_dataset

    def _prepare_group_by_field(self, columns):
        rep_fields = list(AIRRSequenceSet.get_field_type_dict().keys())
        groupby_fields = copy.deepcopy(rep_fields)
        if self.sequence_to_ignore in groupby_fields:
            groupby_fields.remove(self.sequence_to_ignore)
        groupby_fields.remove("duplicate_count")
        groupby_fields.remove("sequence_id")
        groupby_fields.remove("cell_id")

        for field in set(rep_fields).difference(set(columns)):
            if field in groupby_fields:
                groupby_fields.remove(field)

        return groupby_fields

    def _prepare_agg_dict(self, columns, custom_lists):

        agg_dict = {"sequence_id": "first"}

        if self.sequence_to_ignore in columns:
            agg_dict[self.sequence_to_ignore] = "first"

        if "duplicate_count" in columns:
            agg_dict["duplicate_count"] = self.count_agg.value

        if "cell_id" in columns:
            agg_dict["cell_id"] = "first"

        for key in custom_lists:
            agg_dict[key] = "first"

        return agg_dict

    def _process_repertoire(self, repertoire: Repertoire) -> Repertoire:
        no_duplicates = self._process_df(data=repertoire.data.topandas(), custom_lists=list(repertoire.dynamic_fields.keys()))

        no_duplicates.to_csv(f"{self.result_path}/{repertoire.data_filename.stem}_filtered.tsv", sep='\t', index=False)
        write_yaml(Path(f"{self.result_path}/{repertoire.metadata_filename.stem}_filtered.yaml"), repertoire.metadata)

        return Repertoire(data_filename=Path(f"{self.result_path}/{repertoire.data_filename.stem}_filtered.tsv"),
                          metadata_filename=Path(f"{self.result_path}/{repertoire.metadata_filename.stem}_filtered.yaml"),
                          identifier=str(uuid4().hex),
                          dynamic_fields=repertoire.dynamic_fields)

    def _process_df(self, data, custom_lists):
        # presuming default duplicate count = 1 (better than presuming NA)
        # This is important for single-cell datasets where only umi_count is set, and each entry is 1 duplicate
        data['duplicate_count'] = data['duplicate_count'].replace(-1, 1)
        columns = data.columns
        groupby_fields = self._prepare_group_by_field(columns)

        agg_dict = self._prepare_agg_dict(columns, custom_lists)
        no_duplicates = data.groupby(groupby_fields, sort=False).agg(agg_dict).reset_index()

        return no_duplicates


    @staticmethod
    def get_documentation():
        doc = str(DuplicateSequenceFilter.__doc__)

        mapping = {
            "Valid options are defined by the CountAggregationFunction enum.": f"Valid values are: {[e.name.lower() for e in CountAggregationFunction]}.",
            "Valid options are defined by the SequenceType enum.": f"Valid values are: {[e.name.lower() for e in SequenceType]}."
        }

        doc = update_docs_per_mapping(doc, mapping)

        return doc
