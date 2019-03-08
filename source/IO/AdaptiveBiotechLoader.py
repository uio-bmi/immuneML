import glob
import os
import pickle
from multiprocessing.pool import Pool

import pandas as pd

from source.IO.DataLoader import DataLoader
from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.Repertoire import Repertoire
from source.util.PathBuilder import PathBuilder


class AdaptiveBiotechLoader(DataLoader):

    @staticmethod
    def load(path, params: dict = None) -> Dataset:

        files = glob.glob(path + "*." + params["extension"])
        repertoire_filenames, custom_params = AdaptiveBiotechLoader._load_repertoires(files, params)
        dataset = Dataset(filenames=repertoire_filenames, identifier=params["dataset_id"], params=custom_params)

        return dataset

    @staticmethod
    def _load_repertoires(files, params) -> tuple:

        PathBuilder.build(params["result_path"])

        arguments = [(filepath, params) for filepath in files]

        with Pool(params["batch_size"]) as pool:
            output = pool.starmap(AdaptiveBiotechLoader._load_repertoire, arguments)

        repertoire_filenames = [out[0] for out in output]
        #custom_params = AdaptiveBiotechLoader._prepare_custom_params([out[1] for out in output])

        return repertoire_filenames, None

    @staticmethod
    def _load_repertoire(file, params):
        identifier = str(os.path.basename(file).rpartition(".")[0])

        df = pd.read_csv(file, sep="\t", iterator=False,
                         usecols=["v_gene", "j_gene", "amino_acid", "templates", "frame_type"])
        df = df[df.amino_acid.notnull() & df.frame_type.isin(["In"])]
        sequences = df.apply(AdaptiveBiotechLoader._load_sequence, axis=1, args=(params, )).values

        del df

        repertoire = Repertoire(sequences=sequences, metadata=None, identifier=identifier)
        repertoire_filename = params["result_path"] + identifier + ".pickle"

        with open(repertoire_filename, "wb") as f:
            pickle.dump(repertoire, f)

        del sequences
        del repertoire

        return repertoire_filename, None

    @staticmethod
    def _load_sequence(row, params) -> ReceptorSequence:

        chain = "B" if "B" in row["v_gene"] or "B" in row["j_gene"] else "A"
        metadata = SequenceMetadata(v_gene=row["v_gene"], j_gene=row["j_gene"], chain=chain, count=row["templates"],
                                    frame_type=row["frame_type"], region_type=params["region_type"])

        sequence = ReceptorSequence(amino_acid_sequence=row["amino_acid"],
                                    metadata=metadata)

        return sequence
