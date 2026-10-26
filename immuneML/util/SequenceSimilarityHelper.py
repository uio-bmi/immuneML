import logging
import subprocess
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from immuneML.data_model import bnp_util
from immuneML.data_model.SequenceParams import RegionType, Chain
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import SequenceDataset, ReceptorDataset
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.CompAIRRHelper import CompAIRRHelper
from immuneML.util.CompAIRRParams import CompAIRRParams
from immuneML.util.PathBuilder import PathBuilder


class SequenceSimilarityHelper:
    """
    Compares amino acid sequences of sequence and receptor datasets with CompAIRR, ignoring genes and counts, so only
    the sequences themselves are compared.

    Examples are given as a data frame with one column per chain (a single column for sequence datasets). For
    receptors, two examples are within a distance only when every compared chain is, so the distance between two
    receptors is the largest of the distances between their chains.

    CompAIRR allows one insertion or deletion only when the number of differences is 1, so with indels the distance 1
    includes one indel, and a distance of 2 always means two substitutions; sequences of different lengths are never
    within distance 2 of each other.
    """

    @staticmethod
    def get_sequences(dataset: Dataset, region_type: RegionType, chains: List[Chain] = None) -> pd.DataFrame:
        """The amino acid sequences of each example: one column for a sequence dataset, one column per chain (all
        chains of the receptors, or only those listed in chains) for a receptor dataset."""
        sequence_field = bnp_util.get_sequence_field_name(region_type, SequenceType.AMINO_ACID)

        if isinstance(dataset, SequenceDataset):
            assert not chains, \
                (f"{SequenceSimilarityHelper.__name__}: chains can only be chosen for receptor datasets, but dataset "
                 f"{dataset.name} is a sequence dataset.")
            return pd.DataFrame({'sequence': dataset.get_metadata([sequence_field])[sequence_field]})

        elif isinstance(dataset, ReceptorDataset):
            data = dataset.data.topandas()[[sequence_field, 'locus']]
            data['example'] = np.arange(data.shape[0]) // 2
            # loci can be stored as e.g. 'alpha' or 'TRA'; compared as the standard locus names
            data['locus'] = [Chain.get_chain_value(locus) or locus for locus in data['locus']]
            assert not data.duplicated(['example', 'locus']).any(), \
                (f"{SequenceSimilarityHelper.__name__}: some receptors in dataset {dataset.name} have two chains of "
                 f"the same locus, so the chains cannot be told apart.")

            per_chain = data.pivot(index='example', columns='locus', values=sequence_field)
            if chains:
                missing = [chain.value for chain in chains if chain.value not in per_chain.columns]
                assert not missing, \
                    (f"{SequenceSimilarityHelper.__name__}: chains {missing} do not occur in dataset {dataset.name}, "
                     f"which has {list(per_chain.columns)}.")
                per_chain = per_chain[[chain.value for chain in chains]]

            assert not per_chain.isna().any().any(), \
                f"{SequenceSimilarityHelper.__name__}: some receptors in dataset {dataset.name} lack a compared chain."
            return per_chain.reset_index(drop=True)

        else:
            raise NotImplementedError(f"{SequenceSimilarityHelper.__name__}: sequence similarity is defined for "
                                      f"sequence and receptor datasets, got {type(dataset).__name__}.")

    @staticmethod
    def smallest_distance_to_reference(query: Union[pd.Series, pd.DataFrame], reference: Union[pd.Series, pd.DataFrame],
                                       max_edit_distance: int, work_path: Path, compairr_path: str = None,
                                       threads: int = 1, indels: bool = False) -> pd.Series:
        """
        For each query example, the smallest distance to any reference example, from 0 up to max_edit_distance, or NaN
        if it is further than that. Distance 0 is exact equality; CompAIRR is then run once per distance over the
        query examples that have not matched yet, so each example gets the smallest distance at which it matched.
        The result has the index of query.
        """
        query_df, reference_df = SequenceSimilarityHelper._to_frame(query), SequenceSimilarityHelper._to_frame(reference)
        reference_tuples = set(reference_df.itertuples(index=False, name=None))
        distances = np.array([0. if example in reference_tuples else np.nan
                              for example in query_df.itertuples(index=False, name=None)])

        if max_edit_distance > 0 and not reference_df.empty:
            PathBuilder.build(work_path)
            for distance in range(1, max_edit_distance + 1):
                unmatched_positions = np.flatnonzero(np.isnan(distances))
                if unmatched_positions.size == 0:
                    break

                matched = SequenceSimilarityHelper._match_query(query_df.iloc[unmatched_positions], reference_df,
                                                                distance, indels and distance == 1, work_path,
                                                                compairr_path, threads)
                distances[unmatched_positions[matched]] = distance

        return pd.Series(distances, index=query.index)

    @staticmethod
    def similarity_components(sequences: Union[pd.Series, pd.DataFrame], max_edit_distance: int, work_path: Path,
                              compairr_path: str = None, threads: int = 1, indels: bool = False) -> pd.Series:
        """
        The connected components of the graph that joins two examples when they are within max_edit_distance of each
        other (single linkage), so two examples in different components are never that close. Returns the component
        of each example as 'component_<n>', with the index of sequences.
        """
        sequences_df = SequenceSimilarityHelper._to_frame(sequences)
        PathBuilder.build(work_path)

        if sequences_df.shape[1] == 1:
            components = SequenceSimilarityHelper._cluster_single_chain(sequences_df.iloc[:, 0], max_edit_distance,
                                                                        work_path, compairr_path, threads, indels)
        else:
            pairs = SequenceSimilarityHelper._pairs_on_all_chains(sequences_df, sequences_df, max_edit_distance, indels,
                                                                  work_path, compairr_path, threads, "all_vs_all")
            count = sequences_df.shape[0]
            graph = coo_matrix((np.ones(len(pairs)), ([pair[0] for pair in pairs], [pair[1] for pair in pairs])),
                               shape=(count, count))
            components = connected_components(graph, directed=False)[1]

        return pd.Series([f"component_{component}" for component in components], index=sequences.index)

    @staticmethod
    def _to_frame(sequences: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        return sequences.to_frame() if isinstance(sequences, pd.Series) else sequences

    @staticmethod
    def _match_query(query: pd.DataFrame, reference: pd.DataFrame, distance: int, indels: bool, work_path: Path,
                     compairr_path: str, threads: int) -> np.ndarray:
        """Whether each query example is within distance of some reference example."""
        if query.shape[1] == 1:
            reference_path = work_path / "reference_sequences.tsv"
            query_path = work_path / f"query_sequences_d{distance}.tsv"
            SequenceSimilarityHelper._write_compairr_file(reference.iloc[:, 0], reference_path, "reference")
            SequenceSimilarityHelper._write_compairr_file(query.iloc[:, 0], query_path, "query")

            compairr_params = SequenceSimilarityHelper._make_params(compairr_path, distance, indels, threads,
                                                                    f"existence_d{distance}")
            args = CompAIRRHelper.get_cmd_args(compairr_params, [query_path, reference_path], work_path)
            return SequenceSimilarityHelper._run_existence(args, compairr_params, work_path, query.shape[0])
        else:
            pairs = SequenceSimilarityHelper._pairs_on_all_chains(query, reference, distance, indels, work_path,
                                                                  compairr_path, threads, f"query_d{distance}")
            matched_positions = {pair[0] for pair in pairs}
            return np.array([position in matched_positions for position in range(query.shape[0])], dtype=bool)

    @staticmethod
    def _pairs_on_all_chains(first: pd.DataFrame, second: pd.DataFrame, distance: int, indels: bool, work_path: Path,
                             compairr_path: str, threads: int, name: str) -> set:
        """Pairs (position in first, position in second) that are within distance on every chain (column)."""
        pairs = None
        for chain in first.columns:
            chain_name = f"{name}_{chain}"
            first_path, second_path = work_path / f"{chain_name}_1.tsv", work_path / f"{chain_name}_2.tsv"
            SequenceSimilarityHelper._write_compairr_file(first[chain], first_path, "first")
            SequenceSimilarityHelper._write_compairr_file(second[chain], second_path, "second")

            compairr_params = SequenceSimilarityHelper._make_params(compairr_path, distance, indels, threads,
                                                                    f"pairs_{chain_name}")
            pairs_path = work_path / f"compairr_pairs_{chain_name}.tsv"
            args = [str(compairr_params.compairr_path), "-m", "-d", str(distance), "-t", str(threads), "-f", "-g"] \
                + (["-i"] if indels else []) \
                + ["-o", str(work_path / compairr_params.output_filename),
                   "-l", str(work_path / compairr_params.log_filename), "-p", str(pairs_path),
                   str(first_path), str(second_path)]

            logging.info(f"{SequenceSimilarityHelper.__name__}: running CompAIRR with the following arguments: "
                         f"{' '.join(args)}")
            result = subprocess.run(args, capture_output=True, text=True)
            assert pairs_path.is_file(), \
                (f"{SequenceSimilarityHelper.__name__}: CompAIRR did not write the matching pairs ({result.stderr}). "
                 f"See the log file in {work_path} for more information.")

            chain_pairs = pd.read_csv(pairs_path, sep="\t", usecols=["sequence_id_1", "sequence_id_2"])
            chain_pairs = set(zip(chain_pairs["sequence_id_1"].astype(int), chain_pairs["sequence_id_2"].astype(int)))
            pairs = chain_pairs if pairs is None else pairs & chain_pairs

        return pairs

    @staticmethod
    def _cluster_single_chain(sequences: pd.Series, max_edit_distance: int, work_path: Path, compairr_path: str,
                              threads: int, indels: bool) -> list:
        input_path = work_path / "sequences.tsv"
        SequenceSimilarityHelper._write_compairr_file(sequences, input_path, "sequences")

        compairr_params = SequenceSimilarityHelper._make_params(compairr_path, max_edit_distance, indels, threads,
                                                                "clusters")
        args = [str(compairr_params.compairr_path), "-c", "-d", str(max_edit_distance), "-t", str(threads), "-f", "-g"] \
            + (["-i"] if indels else []) \
            + ["-o", str(work_path / compairr_params.output_filename), "-l", str(work_path / compairr_params.log_filename),
               str(input_path)]

        logging.info(f"{SequenceSimilarityHelper.__name__}: running CompAIRR with the following arguments: "
                     f"{' '.join(args)}")
        output_file = CompAIRRHelper.verify_compairr_output_path(subprocess.run(args, capture_output=True, text=True),
                                                                 compairr_params, work_path)
        clusters = pd.read_csv(output_file, sep="\t", dtype={"sequence_id": str})

        cluster_per_position = dict(zip(clusters["sequence_id"], clusters["#cluster_no"]))
        assert len(cluster_per_position) == len(sequences), \
            (f"{SequenceSimilarityHelper.__name__}: CompAIRR assigned {len(cluster_per_position)} of {len(sequences)} "
             f"sequences to a cluster. See the log file in {work_path} for more information.")

        return [cluster_per_position[str(position)] for position in range(len(sequences))]

    @staticmethod
    def _write_compairr_file(sequences: pd.Series, filename: Path, repertoire_id: str):
        """CompAIRR input with genes and counts ignored; sequences are identified by their position."""
        pd.DataFrame({"junction_aa": sequences.values,
                      "repertoire_id": repertoire_id,
                      "sequence_id": [str(position) for position in range(len(sequences))]}).to_csv(filename, sep="\t",
                                                                                                    index=False)

    @staticmethod
    def _make_params(compairr_path, distance: int, indels: bool, threads: int, name: str) -> CompAIRRParams:
        return CompAIRRParams(compairr_path=CompAIRRHelper.determine_compairr_path(compairr_path),
                              keep_compairr_input=True, differences=distance, indels=indels, ignore_counts=True,
                              ignore_genes=True, threads=threads, output_filename=f"compairr_out_{name}.txt",
                              log_filename=f"compairr_log_{name}.txt", output_pairs=False, pairs_filename=None,
                              do_repertoire_overlap=False, do_sequence_matching=True)

    @staticmethod
    def _run_existence(args, compairr_params: CompAIRRParams, work_path: Path, query_count: int) -> np.ndarray:
        logging.info(f"{SequenceSimilarityHelper.__name__}: running CompAIRR with the following arguments: "
                     f"{' '.join(args)}")
        result = CompAIRRHelper.process_compairr_output_file(subprocess.run(args, capture_output=True, text=True),
                                                             compairr_params, work_path)

        # CompAIRR does not have to return the sequences in the order they were given in
        result.index = result.index.astype(str)
        query_ids = [str(position) for position in range(query_count)]
        assert set(result.index) == set(query_ids), \
            (f"{SequenceSimilarityHelper.__name__}: CompAIRR returned {len(set(result.index))} sequence identifiers "
             f"for {query_count} query sequences. See the log file in {work_path} for more information.")

        return (result.reindex(query_ids)["reference"] > 0).values
