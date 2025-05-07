import numpy as np

from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.SequenceParams import RegionType, Chain
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import SequenceDataset, ReceptorDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator


class SillyEncoder(DatasetEncoder):
    """
    This SillyEncoder class is a placeholder for a real encoder.
    It computes a set of random numbers as features for a given dataset.

    **Dataset type:**

    - SequenceDatasets

    - ReceptorDatasets

    - RepertoireDatasets


    **Specification arguments:**

    - random_seed (int): The random seed for generating random features.

    - embedding_len (int): The number of random features to generate per example.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                my_silly_encoder:
                    Silly: # name of the class (without 'Encoder' suffix)
                        random_seed: 1
                        embedding_len: 5
    """

    def __init__(self, random_seed: int, embedding_len: int, name: str = None):
        # The encoder name contains the user-defined name for the encoder. It may be used by some reports.
        super().__init__(name=name)

        # All user parameters are set here.
        # Default parameters must not be defined in the Encoder class, but in a default parameters file.
        self.random_seed = random_seed
        self.embedding_len = embedding_len

    @staticmethod
    def build_object(dataset=None, **params):
        # build_object is called early in the immuneML run, before the analysis takes place.
        # Its purpose is to fail early when a class is called incorrectly (checking parameters and dataset),
        # and provide user-friendly error messages.

        # ParameterValidator contains many utility functions for checking user parameters
        ParameterValidator.assert_type_and_value(params['random_seed'], int, SillyEncoder.__name__, 'random_seed', min_inclusive=1)
        ParameterValidator.assert_type_and_value(params['embedding_len'], int, SillyEncoder.__name__, 'embedding_len', min_inclusive=1, max_inclusive=100)

        # An error should be thrown if the dataset type is incompatible with the Encoder.
        # If different sub-classes are defined for each dataset type (e.g., OneHotRepertoireEncoder),
        # an instance of the dataset-specific class must be returned here.
        if isinstance(dataset, SequenceDataset) or isinstance(dataset, ReceptorDataset) or isinstance(dataset, RepertoireDataset):
            return SillyEncoder(**params)
        else:
            raise ValueError("SillyEncoder is only defined for dataset types SequenceDataset, ReceptorDataset or RepertoireDataset")

    def encode(self, dataset, params: EncoderParams) -> Dataset:
        np.random.seed(self.random_seed)

        # Generate the design matrix from the sequence dataset
        encoded_examples = self._get_encoded_examples(dataset)

        # EncoderHelper contains some utility functions, including this function for encoding the labels
        labels = EncoderHelper.encode_dataset_labels(dataset, params.label_config, params.encode_labels)

        # Each feature is represented by some meaningful name
        feature_names = [f"random_number_{i}" for i in range(self.embedding_len)]

        encoded_data = EncodedData(examples=encoded_examples,
                                   labels=labels,
                                   example_ids=dataset.get_example_ids(),
                                   feature_names=feature_names,
                                   encoding=SillyEncoder.__name__) # When using dataset-specific encoders,
                                                                   # make sure to use the general encoder name here
                                                                   # (e.g., OneHotEncoder.__name__, not OneHotSequenceEncoder.__name__)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = encoded_data

        return encoded_dataset

    def _get_encoded_examples(self, dataset: Dataset) -> np.array:
        if isinstance(dataset, SequenceDataset):
            return self._get_encoded_sequences(dataset)
        elif isinstance(dataset, ReceptorDataset):
            return self._get_encoded_receptors(dataset)
        elif isinstance(dataset, RepertoireDataset):
            return self._get_encoded_repertoires(dataset)

    def _get_encoded_sequences(self, dataset: SequenceDataset) -> np.array:

        # for sequence dataset, the data can be access as a list of ReceptorSequence objects:
        for sequence in dataset.get_data():
            # Each sequence is a ReceptorSequence object.
            # Different properties of the sequence can be retrieved here, examples:
            identifier = sequence.sequence_id
            aa_seq = sequence.sequence_aa
            v_gene = sequence.v_call
            j_gene = sequence.j_call

        # alternatively, data can be accessed through the AIRRSequenceSet object
        # (a bionumpy dataclass, similar to pandas dataframes):

        all_cdr3_aa = dataset.data.cdr3_aa.tolist()  # a list of strings of all sequences in the dataset
        all_v_call = dataset.data.v_call.tolist()  # a list of strings of all v genes in the dataset

        # it can also be accessed as a bionumpy dataclass object:
        cdr3_aa = dataset.data.cdr3_aa  # useful for faster computations

        # in this example, we just return a random matrix of n_examples x n_features
        return np.random.rand(dataset.element_count, self.embedding_len)

    def _get_encoded_receptors(self, dataset: ReceptorDataset) -> np.array:

        # for receptor dataset, the data can be access as a list of Receptor objects:
        for receptor in dataset.get_data():
            # A Receptor contains two paired ReceptorSequence objects
            identifier = receptor.receptor_id
            chain1, chain2 = receptor.get_chains()
            sequence1 = receptor.get_chain(chain1)
            sequence2 = receptor.get_chain(chain2)

            # Properties of the specific ReceptorSequences can be retrieved, examples:
            aa_seq1 = sequence1.sequence_aa # gets the amino acid sequence by default (alternative: nucleotide)
            v_gene_seq1 = sequence1.v_call # gets the v and j genes
            j_gene_seq1 = sequence1.j_call

        # data can be also accessed through the AIRRSequenceSet object
        # (a bionumpy dataclass, similar to pandas dataframes)
        # this is useful when speed is important

        # all cdr3 aa sequences of the beta chain
        cdr3_aa_beta = dataset.data.cdr3_aa[dataset.data.locus == Chain.BETA.to_string()].tolist()
        # all cdr3 aa sequences of the alpha chain
        cdr3_aa_alpha = dataset.data.cdr3_aa[dataset.data.locus == Chain.ALPHA.to_string()].tolist()

        # here we just return a random matrix of n_examples x n_features
        return np.random.rand(dataset.element_count, self.embedding_len)

    def _get_encoded_repertoires(self, dataset: RepertoireDataset) -> np.array:
        encoded_repertoires = []

        for repertoire in dataset.get_data():
            # Each repertoire is a Repertoire object.
            # Receptor sequences from the repertoire objects can be retrieved as objects for the given region type:
            for sequence in repertoire.sequences(region_type=RegionType.IMGT_CDR3):
                assert isinstance(sequence, ReceptorSequence) # each sequence is an object of ReceptorSequence class
                seq_id = sequence.sequence_id
                aa_seq = sequence.sequence_aa
                v_call = sequence.v_call

            # alternatively, the sequences can be retrieved as AIRRSequenceSet objects (bionumpy dataclasses,
            # similar to pandas dataframes):
            all_aa_sequences = repertoire.data.cdr3_aa.tolist()  # a list of strings of all sequences in the repertoire
            all_aa_sequences = repertoire.data.cdr3_aa  # a bionumpy dataclass object containing all sequences in the repertoire

            # other properties can also be accessed in the same way:
            v_calls = repertoire.data.v_call.tolist()  # a list of strings of all v genes in the repertoire

            # In this encoding, repertoire information is ignored, random features are generated
            random_encoding = np.random.rand(self.embedding_len)
            encoded_repertoires.append(random_encoding)

        return np.array(encoded_repertoires)