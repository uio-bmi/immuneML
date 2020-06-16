import copy
from multiprocessing.pool import Pool

import pandas as pd

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.SequenceType import SequenceType
from source.preprocessing.filters.CountAggregationFunction import CountAggregationFunction
from source.preprocessing.filters.Filter import Filter


class DuplicateSequenceFilter(Filter):
    """
    Collapses duplicate nucleotide or amino acid sequences within each repertoire in the given RepertoireDataset.
    This filter can be applied to Repertoires and RepertoireDatasets.

    Sequences are considered duplicates if the following fields are identical:
      - amino acid or nucleotide sequence (whichever is specified)
      - v and j genes (note that the full field including subgroup + gene is used for matching,
                       i.e. V1 and V1-1 are not considered duplicates)
      - chain
      - region type

    For all other fields (the non-specified sequence type, custom lists, sequence identifier) only the first occurring
    value is kept.
    Note that this means the count value of a sequence with a given sequence identifier might not be the same as before
    removing duplicates, unless count_agg = FIRST is used.

    Attributes:

        filter_sequence_type (:py:obj:`~source.environment.SequenceType.SequenceType`): Whether the sequences should be
            collapsed on the nucleotide or amino acid level.
        batch_size (int): number of repertoires that can be loaded at the same time (only affects the speed)
        count_agg (:py:obj:`~source.preprocessing.filters.CountAggregationFunction.CountAggregationFunction`): determines
            how the sequence counts of duplicate sequences are aggregated (for example: summing, max/min values).


    Specification:

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
    def process_repertoire(repertoire: Repertoire, params: dict) -> Repertoire:
        data = pd.DataFrame(repertoire.load_data())

        groupby_fields = copy.deepcopy(Repertoire.FIELDS)
        groupby_fields.remove(params["sequence_to_ignore"])
        groupby_fields.remove("counts")
        groupby_fields.remove("sequence_identifiers")
        groupby_fields.remove("cell_ids")
        groupby_fields.remove("frame_types")

        agg_dict = {"counts": params["count_agg"].value,
                    "sequence_identifiers": "first",
                    params["sequence_to_ignore"]: "first"}

        if "cell_ids" in data.columns:
            agg_dict["cell_ids"] = "first"

        custom_lists = list(set(data.columns) - set(Repertoire.FIELDS))

        for key in custom_lists:
            agg_dict[key] = "first"

        # Chain objects can not be aggregated, convert to strings
        data["chains"] = [chain.value for chain in data["chains"]]

        no_duplicates = data.groupby(groupby_fields).agg(agg_dict).reset_index()

        processed_repertoire = Repertoire.build(sequence_aas=list(no_duplicates["sequence_aas"]),
                                                sequences=list(no_duplicates["sequences"]),
                                                v_genes=list(no_duplicates["v_genes"]),
                                                j_genes=list(no_duplicates["j_genes"]),
                                                chains=[Chain(key) for key in list(no_duplicates["chains"])],
                                                counts=list(no_duplicates["counts"]),
                                                region_types=list(no_duplicates["region_types"]),
                                                custom_lists={key: list(no_duplicates[key]) for key in custom_lists},
                                                sequence_identifiers=list(no_duplicates["sequence_identifiers"]),
                                                path=params["result_path"])

        return processed_repertoire

    def process_dataset(self, dataset: RepertoireDataset, result_path: str) -> RepertoireDataset:
        params = {"result_path": result_path, "filter_sequence_type": self.filter_sequence_type, "count_agg": self.count_agg,
                  "batch_size": self.batch_size, "sequence_of_interest": self.sequence_of_interest,
                  "sequence_to_ignore": self.sequence_to_ignore}

        return DuplicateSequenceFilter.process(dataset, params)
