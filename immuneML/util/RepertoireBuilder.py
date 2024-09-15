import uuid
from pathlib import Path

import pandas as pd

from immuneML.data_model.SequenceSet import ReceptorSequence, Repertoire
from immuneML.data_model.bnp_util import write_yaml
from immuneML.util.PathBuilder import PathBuilder


class RepertoireBuilder:
    """
    Helper class for tests: creates repertoires from a list of a list of sequences and stores them in the given path
    """
    @staticmethod
    def build(sequences: list, path: Path, labels: dict = None, seq_metadata: list = None, subject_ids: list = None):

        if subject_ids is not None:
            assert len(subject_ids) == len(sequences)

        if seq_metadata is not None:
            assert len(sequences) == len(seq_metadata)
            for index, sequence_list in enumerate(sequences):
                assert len(sequence_list) == len(seq_metadata[index])

        assert not (path / "repertoires").is_dir(), f"RepertoireBuilder: attempted to store new repertoires at {path / 'repertoires'} but this folder already exists. " \
                                                    f"Please remove this folder or specify a different path. "

        PathBuilder.build(path)
        rep_path = PathBuilder.build(path / "repertoires")

        repertoires = []
        if subject_ids is None:
            subject_ids = []

        for rep_index, sequence_list in enumerate(sequences):
            rep_sequences = []
            if len(subject_ids) < len(sequences):
                subject_ids.append("rep_" + str(rep_index))

            df = pd.DataFrame({
                'cdr3_aa': sequence_list,
            })

            if seq_metadata is None:
                df['v_call'], df['j_call'], df['duplicate_count'], df['locus'] = "TRBV1-1*01", "TRBJ1-1*01", 1, "TRB"
            else:
                df = pd.concat([df, pd.DataFrame.from_records(seq_metadata[rep_index])], axis=1)

            df.to_csv(str(rep_path / ''))

            if labels is not None:
                metadata = {key: labels[key][rep_index] for key in labels.keys()}
            else:
                metadata = {}

            metadata = {**metadata, **{"subject_id": subject_ids[rep_index]}}
            write_yaml(rep_path / f"_rep_{rep_index}.yaml", metadata)

            repertoire = Repertoire(rep_path / f"rep_{rep_index}.tsv", rep_path / f"_rep_{rep_index}.yaml")
            repertoires.append(repertoire)

        df = pd.DataFrame({**{"filename": [repertoire.data_filename for repertoire in repertoires],
                              "subject_id": subject_ids,
                              "identifier": [repertoire.identifier for repertoire in repertoires]},
                           **(labels if labels is not None else {})})
        df.to_csv(path / "metadata.csv", index=False)

        return repertoires, path / "metadata.csv"
