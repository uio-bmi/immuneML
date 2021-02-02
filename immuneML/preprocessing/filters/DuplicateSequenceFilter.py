import copy
from multiprocessing.pool import Pool
from pathlib import Path

import pandas as pd

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

    Arguments:

        filter_sequence_type (:py:obj:`~immuneML.environment.SequenceType.SequenceType`): Whether the sequences should be collapsed on the nucleotide or amino acid level. Valid options are defined by the SequenceType enum.

        batch_size (int): number of repertoires that can be loaded at the same time (only affects the speed)

        count_agg (:py:obj:`~immuneML.preprocessing.filters.CountAggregationFunction.CountAggregationFunction`): determines how the sequence counts of duplicate sequences are aggregated. Valid options are defined by the CountAggregationFunction enum.

    YAML specification:

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

    def __init__(self, filter_sequence_type: SequenceType, batch_size: int, count_agg: CountAggregationFunction):
        self.filter_sequence_type = filter_sequence_type
        self.count_agg = count_agg
        self.batch_size = batch_size

        self.sequence_of_interest = "sequence_aas" if filter_sequence_type == SequenceType.AMINO_ACID else "sequences"
        self.sequence_to_ignore = "sequences" if self.sequence_of_interest == "sequence_aas" else "sequence_aas"

        assert self.sequence_of_interest in Repertoire.FIELDS
        assert self.sequence_to_ignore in Repertoire.FIELDS

    @staticmethod
    def process(dataset: RepertoireDataset, params: dict) -> RepertoireDataset:
        processed_dataset = copy.deepcopy(dataset)

        with Pool(params["batch_size"]) as pool:
            repertoires = pool.starmap(DuplicateSequenceFilter.process_repertoire,
                                       [(repertoire, params) for repertoire in dataset.repertoires])

        processed_dataset.repertoires = repertoires

        return processed_dataset

    @staticmethod
    def _prepare_group_by_field(params, columns):
        groupby_fields = copy.deepcopy(list(Repertoire.FIELDS))
        groupby_fields.remove(params["sequence_to_ignore"])
        groupby_fields.remove("counts")
        groupby_fields.remove("sequence_identifiers")
        groupby_fields.remove("cell_ids")
        groupby_fields.remove("frame_types")

        for field in set(Repertoire.FIELDS).difference(set(columns)):
            if field in groupby_fields:
                groupby_fields.remove(field)

        return groupby_fields

    @staticmethod
    def _prepare_agg_dict(params, columns, custom_lists):

        agg_dict = {"sequence_identifiers": "first"}

        if params["sequence_to_ignore"] in columns:
            agg_dict[params["sequence_to_ignore"]] = "first"

        if "counts" in columns:
            agg_dict["counts"] = params["count_agg"].value

        if "cell_ids" in columns:
            agg_dict["cell_ids"] = "first"

        for key in custom_lists:
            agg_dict[key] = "first"

        return agg_dict

    @staticmethod
    def process_repertoire(repertoire: Repertoire, params: dict) -> Repertoire:
        data = pd.DataFrame(repertoire.load_data())

        groupby_fields = DuplicateSequenceFilter._prepare_group_by_field(params, data.columns)
        custom_lists = list(set(data.columns) - set(Repertoire.FIELDS))
        agg_dict = DuplicateSequenceFilter._prepare_agg_dict(params, data.columns, custom_lists)

        # Chain objects can not be aggregated, convert to strings
        if "chains" in data.columns:
            data["chains"] = [chain.value if isinstance(chain, Chain) else chain for chain in data["chains"]]
        else:
            data["chains"] = None

        no_duplicates = data.groupby(groupby_fields).agg(agg_dict).reset_index()

        processed_repertoire = Repertoire.build(sequence_aas=list(no_duplicates["sequence_aas"]) if "sequence_aas" in no_duplicates.columns else None,
                                                sequences=list(no_duplicates["sequences"]) if "sequences" in no_duplicates.columns else None,
                                                v_genes=list(no_duplicates["v_genes"]) if "v_genes" in no_duplicates.columns else None,
                                                j_genes=list(no_duplicates["j_genes"]) if 'j_genes' in no_duplicates.columns else None,
                                                chains=[Chain(key) for key in list(no_duplicates["chains"])] if "chains" in no_duplicates.columns else None,
                                                counts=list(no_duplicates["counts"]) if "counts" in no_duplicates else None,
                                                region_types=list(no_duplicates["region_types"]) if "region_types" in no_duplicates else None,
                                                custom_lists={key: list(no_duplicates[key]) for key in custom_lists},
                                                sequence_identifiers=list(no_duplicates["sequence_identifiers"]),
                                                metadata=copy.deepcopy(repertoire.metadata),
                                                path=params["result_path"],
                                                filename_base=f"{repertoire.data_filename.stem}_filtered")

        return processed_repertoire

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path) -> RepertoireDataset:
        params = {"result_path": result_path, "filter_sequence_type": self.filter_sequence_type, "count_agg": self.count_agg,
                  "batch_size": self.batch_size, "sequence_of_interest": self.sequence_of_interest,
                  "sequence_to_ignore": self.sequence_to_ignore}

        return DuplicateSequenceFilter.process(dataset, params)


    @staticmethod
    def get_documentation():
        doc = str(DuplicateSequenceFilter.__doc__)

        mapping = {
            "Valid options are defined by the CountAggregationFunction enum.": f"Valid values are: {[e.name for e in CountAggregationFunction]}.",
            "Valid options are defined by the SequenceType enum.": f"Valid values are: {[e.name for e in SequenceType]}."
        }

        doc = update_docs_per_mapping(doc, mapping)

        return doc
