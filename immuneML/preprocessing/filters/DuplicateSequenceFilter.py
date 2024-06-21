import copy
from dataclasses import fields as get_fields
from multiprocessing.pool import Pool
from pathlib import Path
from uuid import uuid4

import pandas as pd

from immuneML.data_model.bnp_util import write_yaml
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.preprocessing.filters.CountAggregationFunction import CountAggregationFunction
from immuneML.preprocessing.filters.Filter import Filter
from immuneML.util.ParameterValidator import ParameterValidator
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

    - batch_size (int): number of repertoires that can be loaded at the same time (only affects the speed)

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

    """

    @classmethod
    def build_object(cls, **kwargs):
        location = cls.__name__
        ParameterValidator.assert_keys(kwargs.keys(), ["filter_sequence_type", "batch_size", "count_agg"], location,
                                       "DuplicateSequenceFilter")
        ParameterValidator.assert_in_valid_list(kwargs["filter_sequence_type"].upper(), [item.name for item in SequenceType],
                                                location, "filter_sequence_type")
        ParameterValidator.assert_in_valid_list(kwargs["count_agg"].upper(), [item.name for item in CountAggregationFunction], location,
                                                "count_agg")
        ParameterValidator.assert_type_and_value(kwargs["batch_size"], int, location, "batch_size", 1)
        return DuplicateSequenceFilter(filter_sequence_type=SequenceType[kwargs["filter_sequence_type"].upper()],
                                       batch_size=kwargs["batch_size"], count_agg=CountAggregationFunction[kwargs["count_agg"].upper()])

    def __init__(self, filter_sequence_type: SequenceType, batch_size: int, count_agg: CountAggregationFunction, result_path: Path = None):
        super().__init__(result_path)
        self.filter_sequence_type = filter_sequence_type
        self.count_agg = count_agg
        self.batch_size = batch_size

        self.sequence_of_interest = "sequence_aa" if filter_sequence_type == SequenceType.AMINO_ACID else "sequence"
        self.sequence_to_ignore = "sequence" if self.sequence_of_interest == "sequence_aa" else "sequence_aa"

        assert self.sequence_of_interest in Repertoire.FIELDS, f"{DuplicateSequenceFilter.__name__}: {self.sequence_of_interest} not in {Repertoire.FIELDS}"
        assert self.sequence_to_ignore in Repertoire.FIELDS, f"{DuplicateSequenceFilter.__name__}: {self.sequence_of_interest} not in {Repertoire.FIELDS}"

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path, number_of_processes=1) -> RepertoireDataset:
        self.result_path = result_path if result_path is not None else self.result_path

        self.check_dataset_type(dataset, [RepertoireDataset], "DuplicateSequenceFilter")

        processed_dataset = copy.deepcopy(dataset)

        with Pool(self.batch_size) as pool:
            repertoires = pool.map(self._process_repertoire, dataset.repertoires)

        processed_dataset.repertoires = repertoires

        return processed_dataset

    def _prepare_group_by_field(self, columns):
        groupby_fields = copy.deepcopy(list(Repertoire.FIELDS))
        groupby_fields.remove(self.sequence_to_ignore)
        groupby_fields.remove("duplicate_count")
        groupby_fields.remove("sequence_id")
        groupby_fields.remove("cell_id")
        groupby_fields.remove("frame_type")

        for field in set(Repertoire.FIELDS).difference(set(columns)):
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
        data = repertoire.load_bnp_data().topandas()
        data['duplicate_count'] = [el if el != -1 else pd.NA for el in data['duplicate_count']]
        columns = data.columns

        groupby_fields = self._prepare_group_by_field(columns)
        custom_lists = list(set(columns) - set(Repertoire.FIELDS))
        agg_dict = self._prepare_agg_dict(columns, custom_lists)

        # Chain objects can not be aggregated, convert to strings
        if "chain" in columns:
            data["chain"] = data.chain.tolist()
        else:
            data["chain"] = None

        no_duplicates = data.groupby(groupby_fields, sort=False).agg(agg_dict).reset_index()

        no_duplicates.to_csv(f"{self.result_path}/{repertoire.data_filename.stem}_filtered.tsv", sep='\t', index=False)
        write_yaml(Path(f"{self.result_path}/{repertoire.metadata_filename.stem}_filtered.yaml"), repertoire.metadata)

        return Repertoire(Path(f"{self.result_path}/{repertoire.data_filename.stem}_filtered.tsv"),
                          Path(f"{self.result_path}/{repertoire.metadata_filename.stem}_filtered.yaml"),
                          str(uuid4().hex))

    @staticmethod
    def get_documentation():
        doc = str(DuplicateSequenceFilter.__doc__)

        mapping = {
            "Valid options are defined by the CountAggregationFunction enum.": f"Valid values are: {[e.name.lower() for e in CountAggregationFunction]}.",
            "Valid options are defined by the SequenceType enum.": f"Valid values are: {[e.name.lower() for e in SequenceType]}."
        }

        doc = update_docs_per_mapping(doc, mapping)

        return doc
