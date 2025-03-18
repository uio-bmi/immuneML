import copy
import os
from multiprocessing.pool import Pool
from pathlib import Path
from uuid import uuid4

import pandas as pd

from immuneML.data_model import bnp_util
from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.bnp_util import write_yaml, read_yaml
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.preprocessing.filters.CountAggregationFunction import CountAggregationFunction
from immuneML.preprocessing.filters.Filter import Filter
from immuneML.util.ParameterValidator import ParameterValidator
from scripts.specification_util import update_docs_per_mapping


class DuplicateSequenceFilter(Filter):
    """
    Collapses duplicate nucleotide or amino acid sequences within each repertoire in the given RepertoireDataset or within a SequenceDataset.
    This filter can be applied to Repertoires, RepertoireDatasets, and SequenceDatasets.

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
                        batch_size: 4
                        count_agg: SUM
                        region_type: IMGT_CDR3

    """

    @classmethod
    def build_object(cls, **kwargs):
        location = cls.__name__
        ParameterValidator.assert_keys(kwargs.keys(), ["filter_sequence_type", "batch_size", "count_agg", "region_type"], location,
                                       "DuplicateSequenceFilter")
        ParameterValidator.assert_in_valid_list(kwargs["filter_sequence_type"].upper(), [item.name for item in SequenceType],
                                                location, "filter_sequence_type")
        ParameterValidator.assert_in_valid_list(kwargs["region_type"].upper(), [item.name for item in RegionType],
                                                location, "region_type")
        ParameterValidator.assert_in_valid_list(kwargs["count_agg"].upper(), [item.name for item in CountAggregationFunction], location,
                                                "count_agg")
        ParameterValidator.assert_type_and_value(kwargs["batch_size"], int, location, "batch_size", 1)
        return DuplicateSequenceFilter(filter_sequence_type=SequenceType[kwargs["filter_sequence_type"].upper()],
                                       region_type=RegionType[kwargs['region_type'].upper()],
                                       batch_size=kwargs["batch_size"], count_agg=CountAggregationFunction[kwargs["count_agg"].upper()])

    def __init__(self, filter_sequence_type: SequenceType, batch_size: int, count_agg: CountAggregationFunction,
                 result_path: Path = None, region_type: RegionType = RegionType.IMGT_CDR3):
        super().__init__(result_path)
        self.filter_sequence_type = filter_sequence_type
        self.region_type = region_type
        self.count_agg = count_agg
        self.batch_size = batch_size

        self.sequence_of_interest = bnp_util.get_sequence_field_name(self.region_type, self.filter_sequence_type)
        self.sequence_to_ignore = bnp_util.get_sequence_field_name(self.region_type, [t for t in SequenceType if t != self.filter_sequence_type][0])

    def process_dataset(self, dataset, result_path: Path, number_of_processes=1):
        self.result_path = result_path if result_path is not None else self.result_path

        self.check_dataset_type(dataset, [RepertoireDataset, SequenceDataset], "DuplicateSequenceFilter")

        processed_dataset = dataset.clone()

        if isinstance(dataset, RepertoireDataset):
            with Pool(self.batch_size) as pool:
                repertoires = pool.map(self._process_repertoire, dataset.repertoires)

            processed_dataset.repertoires = repertoires
            return processed_dataset

        elif isinstance(dataset, SequenceDataset):
            processed_dataset = self._process_sequences(dataset)
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
        no_duplicates = self._filter_duplicates(repertoire)

        no_duplicates.to_csv(f"{self.result_path}/{repertoire.data_filename.stem}_filtered.tsv", sep='\t', index=False)
        write_yaml(Path(f"{self.result_path}/{repertoire.metadata_filename.stem}_filtered.yaml"), repertoire.metadata)

        return Repertoire(data_filename=Path(f"{self.result_path}/{repertoire.data_filename.stem}_filtered.tsv"),
                          metadata_filename=Path(f"{self.result_path}/{repertoire.metadata_filename.stem}_filtered.yaml"),
                          identifier=str(uuid4().hex),
                          dynamic_fields=repertoire.dynamic_fields)

    def _process_sequences(self, dataset: SequenceDataset) -> SequenceDataset:
        no_duplicates = self._filter_duplicates(dataset)
        metadata = read_yaml(dataset.dataset_file)

        os.makedirs(self.result_path, exist_ok=True)
        no_duplicates.to_csv(f"{self.result_path}/{dataset.filename.stem}_filtered.tsv", sep='\t', index=False)
        write_yaml(Path(f"{self.result_path}/{dataset.dataset_file.stem}_filtered.yaml"), metadata)

        return SequenceDataset(name=dataset.name,
                               filename=Path(f"{self.result_path}/{dataset.filename.stem}_filtered.tsv"),
                               dataset_file=Path(f"{self.result_path}/{dataset.dataset_file.stem}_filtered.yaml"),
                               dynamic_fields=dataset.dynamic_fields)

    def _filter_duplicates(self, dataset):
        data = dataset.data.topandas()
        data['duplicate_count'].replace(-1, pd.NA, inplace=True)
        columns = data.columns

        groupby_fields = self._prepare_group_by_field(columns)
        custom_lists = list(dataset.dynamic_fields.keys())
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
