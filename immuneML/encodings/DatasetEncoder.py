import abc
import pickle
import shutil
from pathlib import Path
from typing import List

from immuneML.IO.dataset_export.ImmuneMLExporter import ImmuneMLExporter
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.encodings.EncoderParams import EncoderParams


class DatasetEncoder(metaclass=abc.ABCMeta):
    """

    **YAML specification:**

        encodings:
            e1: <encoder_class> # encoding without parameters

            e2:
                <encoder_class>: # encoding with parameters
                    parameter: value
    """
    def __init__(self, name: str = None):
        self.name = name

    @staticmethod
    @abc.abstractmethod
    def build_object(dataset: Dataset, **params):
        """
        Creates an instance of the relevant subclass of the DatasetEncoder class using the given parameters.
        This method will be called during parsing time (early in the immuneML run), such that parameters and dataset type can be tested here.

        The build_object method should do the following:

          1. Check parameters: immuneML should crash if wrong user parameters are specified. The ParameterValidator utility class may be used for parameter testing.

          2. Check the dataset type: immuneML should crash if the wrong dataset type is specified for this encoder. For example, DeepRCEncoder should only work for RepertoireDatasets and crash if the dataset is of another type.

          3. Create an instance of the correct Encoder class, using the given parameters. Return this object.
             Some encoders have different subclasses depending on the dataset type. Make sure to return an instance of the correct subclass.
             For instance: KmerFrequencyEncoder has different subclasses for each dataset type. When the dataset is a Repertoire dataset, KmerFreqRepertoireEncoder should be returned.


        Args:

            **params: keyword arguments that will be provided by users in the specification (if immuneML is used as a command line tool) or in the
             dictionary when calling the method from the code, and which should be used to create the Encoder object

        Returns:

            the object of the appropriate Encoder class

        """
        pass

    @abc.abstractmethod
    def encode(self, dataset, params: EncoderParams) -> Dataset:
        """
        This is the main encoding method of the Encoder. It takes in a given dataset, computes an EncodedData object,
        and returns a copy of the dataset with the attached EncodedData object.


        Args:

            dataset: A dataset object (Sequence, Receptor or RepertoireDataset)

            params: An EncoderParams object containing few utility parameters which may be used during encoding (e.g., number of parallel processes to use).

        Returns:

            A copy of the original dataset, with an EncodedData object added to the dataset.encoded_data field.

        """

        pass

    @staticmethod
    def load_encoder(encoder_file: Path):
        """
        The load_encoder method can load the encoder given the folder where the same class of the model was previously stored using the store function.
        Encoders are stored in pickle format. If the encoder uses additional files, they should be explicitly loaded here as well.

        If there are no additional files, this method does not need to be overwritten.
        If there are additional files, its contents should be as follows:

            encoder = DatasetEncoder.load_encoder(encoder_file)
            encoder.my_additional_file = DatasetEncoder.load_attribute(encoder, encoder_file, "my_additional_file")


        Arguments:

            encoder_file (Path): path to the encoder file where the encoder was stored using store() function

        Returns:

            the loaded Encoder object
        """
        with encoder_file.open("rb") as file:
            encoder = pickle.load(file)
        return encoder

    @staticmethod
    def load_attribute(encoder, encoder_file: Path, attribute: str):
        """
        Utility method for loading correct file paths when loading an encoder (see: load_encoder).
        This method should not be overwritten.
        """
        if encoder_file is not None:
            file_path = encoder_file.parent / getattr(encoder, attribute).name
            setattr(encoder, attribute, file_path)
            assert getattr(encoder, attribute).is_file(), f"{type(encoder).__name__}: could not load {attribute} from {getattr(encoder, attribute)}."
        return encoder

    @staticmethod
    def store_encoder(encoder, encoder_file: Path):
        """
        The store_encoder function stores the given encoder such that it can be imported later using load function.
        It uses pickle to store the Python object, as well as the additional filenames which should be returned by
        the get_additional_files() method.

        This method should not be overwritten.


        Arguments:

            encoder: the encoder object

            encoder_file (Path): path to the encoder file

        Returns:

            the encoder file

        """
        with encoder_file.open("wb") as file:
            pickle.dump(encoder, file)

        encoder_dir = encoder_file.parent
        for file in encoder.get_additional_files():
            shutil.copy(file, encoder_dir / file.name)

        return encoder_file

    @staticmethod
    def get_additional_files() -> List[str]:
        """
        Should return a list with all the files that need to be stored when storing the encoder.
        For example, SimilarToPositiveSequenceEncoder stores all 'positive' sequences in the training data,
        and predicts a sequence to be 'positive' if it is similar to any positive sequences in the training data.
        In that case, these positive sequences are stored in a file.

        For many encoders, it may not be necessary to store additional files.
        """
        return []

    def set_context(self, context: dict):
        """
        This method can be used to attach the full dataset (as part of a dictionary), as opposed to the dataset which
        is passed to the .encode() method. When training ML models, that data split is usually a training/validation
        subset of the total dataset.

        In most cases, an encoder should only use the 'dataset' argument passed to the .encode() method to compute
        the encoded data. Using information from the full dataset, which includes the test data, may result in data leakage.
        For example, some encoders normalise the computed feature values (e.g., KmerFrequencyEncoder). Such normalised
        feature values should be based only on the current data split, and test data should remain unseen.

        To avoid confusion about which version of the dataset to use, the full dataset is by default not attached,
        and attaching the full dataset should be done explicitly when required. For instance, if the encoded data is
        some kind of distance matrix (e.g., DistanceEncoder), the distance between examples in the training and test
        dataset should be included. Note that this does *not* entail data leakage: the test examples are not used to
        improve the computation of distances. The distances to test examples are determined by an algorithm which does
        not 'learn' from test data.

        To explicitly enable using the full dataset in the encoder, the contents of this method should be as follows:

        self.context = context
        return self


        Args:
            context: a dictionary containing the full dataset
        """
        return self

    def store(self, encoded_dataset, params: EncoderParams):
        """
        Stores the given encoded dataset using the ImmuneMLExporter. This method should not be overwritten.
        """
        ImmuneMLExporter.export(encoded_dataset, params.result_path)
