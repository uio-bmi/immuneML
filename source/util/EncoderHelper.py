import pickle

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.EncoderParams import EncoderParams
from source.util.PathBuilder import PathBuilder


class EncoderHelper:

    @staticmethod
    def prepare_training_ids(dataset: RepertoireDataset, params: EncoderParams):
        PathBuilder.build(params["result_path"])
        if params["learn_model"]:
            train_repertoire_ids = dataset.get_repertoire_ids()
            with open(params["result_path"] + "repertoire_ids.pickle", "wb") as file:
                pickle.dump(train_repertoire_ids, file)
        else:
            with open(params["result_path"] + "repertoire_ids.pickle", "rb") as file:
                train_repertoire_ids = pickle.load(file)
        return train_repertoire_ids
