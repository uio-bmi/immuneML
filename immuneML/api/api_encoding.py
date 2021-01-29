import random
from pathlib import Path

import pandas as pd

from immuneML.IO.dataset_import.MiXCRImport import MiXCRImport
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.encoding_reports.DesignMatrixExporter import DesignMatrixExporter
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.steps.DataEncoder import DataEncoder
from immuneML.workflows.steps.DataEncoderParams import DataEncoderParams


def encode_dataset_by_kmer_freq(path_to_dataset_directory: str, result_path: str, metadata_path: str = None):
    """
    encodes the repertoire dataset using KmerFrequencyEncoder

    Arguments:
        path_to_dataset_directory (str): path to directory containing all repertoire files with .tsv extension in MiXCR format
        result_path (str): where to store the results
        metadata_path(str): csv file with columns "filename", "subject_id", "disease" which is filled by default if value of argument is None,
            otherwise any metadata csv file passed to the function, must include filename and subject_id columns, and an arbitrary disease column
    Returns:
         encoded dataset with encoded data in encoded_dataset.encoded_data.examples
    """
    path_to_dataset_directory = Path(path_to_dataset_directory)
    result_path = Path(result_path)

    if metadata_path is None:
        metadata_path = generate_random_metadata(path_to_dataset_directory, result_path)
    else:
        metadata_path = Path(metadata_path)

    loader = MiXCRImport()
    dataset = loader.import_dataset({
        "is_repertoire": True,
        "path": path_to_dataset_directory,
        "metadata_file": metadata_path,
        "region_type": "IMGT_CDR3",  # import_dataset in only cdr3
        "number_of_processes": 4,  # number of parallel processes for loading the data
        "result_path": result_path,
        "separator": "\t",
        "columns_to_load": ["cloneCount", "allVHitsWithScore", "allJHitsWithScore", "aaSeqCDR3", "nSeqCDR3"],
        "column_mapping": {
            "cloneCount": "counts",
            "allVHitsWithScore": "v_genes",
            "allJHitsWithScore": "j_genes"
        },
    }, "mixcr_dataset")

    label_name = list(dataset.labels.keys())[0]  # label that can be used for ML prediction - by default: "disease" with values True/False

    encoded_dataset = DataEncoder.run(DataEncoderParams(dataset, KmerFrequencyEncoder.build_object(dataset, **{
        "normalization_type": "relative_frequency",  # encode repertoire by the relative frequency of k-mers in repertoire
        "reads": "unique",  # count each sequence only once, do not use clonal count
        "k": 2,  # k-mer length
        "sequence_encoding": "continuous_kmer"  # split each sequence in repertoire to overlapping k-mers
    }), EncoderParams(result_path=result_path,
                      label_config=LabelConfiguration([Label(label_name, dataset.labels[label_name])])), False))

    dataset_exporter = DesignMatrixExporter(dataset=encoded_dataset,
                                            result_path=result_path / "csv_exported", file_format='csv')
    dataset_exporter.generate_report()

    return encoded_dataset


def generate_random_metadata(path_to_dataset_directory: Path, result_path: Path):

    repertoire_filenames = list(path_to_dataset_directory.glob("*"))

    repertoire_count = len(repertoire_filenames)

    df = pd.DataFrame({"filename": [filename.name for filename in repertoire_filenames],
                       "disease": [random.choice([True, False]) for i in range(repertoire_count)],
                       "subject_id": [str(i) for i in range(1, repertoire_count + 1)]})

    PathBuilder.build(result_path)
    metadata_path = result_path / "metadata.csv"
    df.to_csv(metadata_path, index=None)

    return metadata_path
