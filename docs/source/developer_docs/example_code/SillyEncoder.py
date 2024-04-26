import numpy as np

from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator


class SillyEncoder(DatasetEncoder):
    """
    This SillyEncoder class is a placeholder for a real encoder.
    It computes a set of random numbers as features for a given dataset.


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
        encoded_examples = self._get_encoded_examples(dataset, params)

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

        return self._construct_encoded_dataset(dataset, encoded_data)


    def _get_encoded_examples(self, dataset: Dataset, params: EncoderParams) -> np.array:
        if isinstance(dataset, SequenceDataset):
            return self._get_encoded_sequences(dataset, params)
        elif isinstance(dataset, ReceptorDataset):
            return self._get_encoded_receptors(dataset, params)
        elif isinstance(dataset, RepertoireDataset):
            return self._get_encoded_repertoires(dataset, params)

    def _get_encoded_sequences(self, dataset: SequenceDataset, params: EncoderParams) -> np.array:
        encoded_sequences = []

        for sequence in dataset.get_data(params.pool_size):
            # Each sequence is a ReceptorSequence object.
            # Different properties of the sequence can be retrieved here, examples:
            identifier = sequence.get_id()
            aa_seq = sequence.get_sequence() # gets the amino acid sequence by default (alternative: nucleotide)
            v_gene = sequence.get_attribute("v_gene") # gets the v and j genes (without *allele)
            j_gene = sequence.get_attribute("j_gene")

            # In this encoding, sequence information is ignored, random features are generated
            random_encoding = np.random.rand(self.embedding_len)
            encoded_sequences.append(random_encoding)

        return np.array(encoded_sequences)

    def _get_encoded_receptors(self, dataset: ReceptorDataset, params: EncoderParams) -> np.array:
        encoded_receptors = []

        for receptor in dataset.get_data(params.pool_size):
            # Each receptor is a Receptor subclass object (e.g., TCABReceptor, BCReceptor)
            # A Receptor contains two paired ReceptorSequence objects
            identifier = receptor.get_id()
            chain1, chain2 = receptor.get_chains()
            sequence1 = receptor.get_chain(chain1)
            sequence2 = receptor.get_chain(chain2)

            # Properties of the specific ReceptorSequences can be retrieved, examples:
            aa_seq1 = sequence1.get_sequence() # gets the amino acid sequence by default (alternative: nucleotide)
            v_gene_seq1 = sequence1.get_attribute("v_gene") # gets the v and j genes (without *allele)
            j_gene_seq1 = sequence1.get_attribute("j_gene")

            # It's also possible to retrieve this information for both chains at the Receptor level:
            aa_seq1, aa_seq2 = receptor.get_attribute("sequence_aa")
            v_gene_seq1, v_gene_seq2 = receptor.get_attribute("v_gene")

            # In this encoding, sequence information is ignored, random features are generated
            random_encoding = np.random.rand(self.embedding_len)
            encoded_receptors.append(random_encoding)

        return np.array(encoded_receptors)

    def _get_encoded_repertoires(self, dataset: RepertoireDataset, params: EncoderParams) -> np.array:
        encoded_repertoires = []

        for repertoire in dataset.get_data(params.pool_size):
            # Each repertoire is a Repertoire object.
            # Different properties of the repertoire can be retrieved here, examples:
            identifiers = repertoire.get_sequence_identifiers(as_list=True)
            aa_sequences = repertoire.get_sequence_aas(as_list=True)
            v_genes = repertoire.get_v_genes() # gets the v and j genes (without *allele)
            j_genes = repertoire.get_j_genes()
            sequence_counts = repertoire.get_counts()
            chains = repertoire.get_chains()

            # In this encoding, repertoire information is ignored, random features are generated
            random_encoding = np.random.rand(self.embedding_len)
            encoded_repertoires.append(random_encoding)

        return np.array(encoded_repertoires)

    def _construct_encoded_dataset(self, dataset, encoded_data) -> Dataset:
        if isinstance(dataset, SequenceDataset):
            return SequenceDataset(filenames=dataset.get_filenames(),
                                   encoded_data=encoded_data,
                                   labels=dataset.labels,
                                   file_size=dataset.file_size,
                                   dataset_file=dataset.dataset_file)
        elif isinstance(dataset, ReceptorDataset):
            return ReceptorDataset(filenames=dataset.get_filenames(),
                                   encoded_data=encoded_data,
                                   labels=dataset.labels,
                                   file_size=dataset.file_size,
                                   dataset_file=dataset.dataset_file)
        elif isinstance(dataset, RepertoireDataset):
            return RepertoireDataset(repertoires=dataset.repertoires,
                                     encoded_data=encoded_data,
                                     labels=dataset.labels,
                                     metadata_file=dataset.metadata_file)


