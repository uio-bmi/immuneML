How to add a new encoding
===========================

In this tutorial, we will add a new encoder to represent repertoire datasets by k-mer frequencies.

Adding a new encoder
---------------------

To add a new encoder:

1. Add a new package to :py:obj:`~immuneML.encodings` package called "my_kmer_encoding"
2. Add a new class called :code:`MyKmerFrequencyEncoder` to the package. In the YAML specification, the class name will be used without the 'Encoder' suffix.
3. Set :py:obj:`~immuneML.encodings.DatasetEncoder.DatasetEncoder` as a base class to :code:`MyKmerFrequencyEncoder`.
4. Implement the abstract methods :code:`encode()` and :code:`build_object()`.
5. Implement methods to import and export an encoder: :code:`get_additional_files()`, :code:`export_encoder()` and :code:`load_encoder()`, mostly relying on functionality already available in :py:obj:`~immuneML.encodings.DatasetEncoder.DatasetEncoder`.
6. Add class documentation including: what the encoder does, what the arguments are and an example on how to use it from YAML specification.

An example of the implementation of :code:`MyKmerFrequencyEncoder` for the :py:obj:`~immuneML.data_model.dataset.RepertoireDataset.RepertoireDataset` is shown.

.. code-block:: python

    import pickle
    from collections import Counter

    from sklearn.feature_extraction import DictVectorizer
    import numpy as np

    from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
    from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
    from immuneML.data_model.encoded_data.EncodedData import EncodedData
    from immuneML.encodings.DatasetEncoder import DatasetEncoder
    from immuneML.encodings.EncoderParams import EncoderParams
    from immuneML.encodings.preprocessing.FeatureScaler import FeatureScaler
    from immuneML.util.ParameterValidator import ParameterValidator
    from immuneML.util.PathBuilder import PathBuilder


    class MyKmerFrequencyEncoder(DatasetEncoder):
        """
        Encodes the repertoires of the dataset by k-mer frequencies and normalizes the frequencies to zero mean and unit variance.

        Arguments:

            k (int): k-mer length

        YAML specification:

        .. indent with spaces
        .. code-block:: yaml

            my_encoder: # user-defined name in the specs, here it will be 'my_encoder'
                MyKmerFrequency: # name of the class (without 'Encoder' suffix)
                    k: 3 # argument value

        """

        @staticmethod
        def build_object(dataset, **kwargs): # called when parsing YAML, check all user-defined arguments here
            ParameterValidator.assert_keys(kwargs.keys(), ['k', 'name'], MyKmerFrequencyEncoder.__name__, 'KmerFrequency')
            ParameterValidator.assert_type_and_value(kwargs['name'], str, MyKmerFrequencyEncoder.__name__, 'name')
            ParameterValidator.assert_type_and_value(kwargs['k'], int, MyKmerFrequencyEncoder.__name__, 'k', 1, 10)
            ParameterValidator.assert_type_and_value(dataset, RepertoireDataset, MyKmerFrequencyEncoder.__name__, f'dataset under {kwargs["name"]}')

            return MyKmerFrequencyEncoder(**kwargs)

        def __init__(self, k: int, name: str = None):
            # user-defined parameters
            self.k = k  # defined from specs
            self.name = name  # set at runtime by the platform from the key set by user in the specs

            # internal: not seen by the user
            self.scaler_path = None
            self.vectorizer_path = None

        def encode(self, dataset, params: EncoderParams): # called at runtime by the platform
            encoded_repertoires = self._encode_repertoires(dataset, params)
            labels = self._prepare_labels(dataset, params)
            encoded_data = EncodedData(encoded_repertoires["examples"], labels, dataset.get_example_ids(),
                                       encoded_repertoires['feature_names'], encoding=MyKmerFrequencyEncoder.__name__)
            encoded_dataset = RepertoireDataset(repertoires=dataset.repertoires, encoded_data=encoded_data,
                                                labels=dataset.labels, metadata_file=dataset.metadata_file)

            return encoded_dataset

        def _prepare_labels(self, dataset: RepertoireDataset, params: EncoderParams) -> dict:
            """returns a dict in the format {label_name: [label_value_repertoire_1, ..., label_value_repertoire_n]}"""
            return dataset.get_metadata(params.label_config.get_labels_by_name())

        def _encode_repertoires(self, dataset: RepertoireDataset, params: EncoderParams) -> dict:

            examples = self._create_kmer_counts(dataset)
            vectorized = self._vectorize_encoded_examples(examples, params)
            scaled_examples = self._scale_vectorized_examples(vectorized['examples'], params)

            return {"examples": scaled_examples, "feature_names": vectorized["feature_names"]}

        def _create_kmer_counts(dataset: RepertoireDataset):
            examples = []
            for repertoire in dataset.repertoires:
                counter = Counter()
                for sequence in repertoire.sequences:
                    kmers = self._split_sequence_to_kmers(sequence.amino_acid_sequence)
                    counter += Counter(kmers)
                examples.append(counter)

            return examples

        def _scale_vectorized_examples(self, vectorized_examples: np.ndarray, params: EncoderParams) -> np.ndarray:
            self.scaler_path = params.result_path / 'scaler.pickle' if self.scaler_path is None else self.scaler_path

            normalized_examples = FeatureScaler.normalize(vectorized_examples, NormalizationType.RELATIVE_FREQUENCY)
            scaled_examples = FeatureScaler.standard_scale(self.scaler_path, normalized_examples, with_mean=True)

            return scaled_examples

        def _vectorize_encoded_examples(self, examples: list, params: EncoderParams) -> dict:

            if self.vectorizer_path is None:
                self.vectorizer_path = params.result_path / "vectorizer.pickle"

            if params.learn_model:
                vectorizer = DictVectorizer(sparse=False, dtype=float)
                vectorized_examples = vectorizer.fit_transform(examples)
                PathBuilder.build(params.result_path)
                with self.vectorizer_path.open('wb') as file:
                    pickle.dump(vectorizer, file)
            else:
                with self.vectorizer_path.open('rb') as file:
                    vectorizer = pickle.load(file)
                vectorized_examples = vectorizer.transform(examples)

            return {"examples": vectorized_examples, "feature_names": vectorizer.get_feature_names()}

        def _split_sequence_to_kmers(self, sequence: str):
            kmers = []
            for i in range(0, len(sequence) - self.k + 1):
                kmers.append(sequence[i:i + self.k])
            return kmers

        def get_additional_files(self) -> List[str]:
            """Returns a list of files used for encoding"""
            files = []
            if self.scaler_path is not None and self.scaler_path.is_file():
                files.append(self.scaler_path)
            if self.vectorizer_path is not None and self.vectorizer_path.is_file():
                files.append(self.vectorizer_path)
            return files

        @staticmethod
        def export_encoder(path: Path, encoder) -> Path:
            encoder_file = DatasetEncoder.store_encoder(encoder, path / "encoder.pickle")
            return encoder_file

        @staticmethod
        def load_encoder(encoder_file: Path):
            encoder = DatasetEncoder.load_encoder(encoder_file)
            for attribute in ['scaler_path', 'vectorizer_path']:
                encoder = DatasetEncoder.load_attribute(encoder, encoder_file, attribute)
            return encoder

Testing the new encoder
-----------------------

To test the new encoder:

1. Create a package :code:`~test.encodings.my_kmer_encoding`.
2. In the package, create a class :code:`TestMyKmerFrequencyEncoder` that inherits :code:`unittest.TestCase`.
3. Implement test functions as needed.

A test example for TestMyKmerFrequencyEncoder is shown below.

.. code-block:: python

    import os
    import shutil
    from unittest import TestCase

    import numpy as np

    from immuneML.caching.CacheType import CacheType
    from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
    from immuneML.encodings.EncoderParams import EncoderParams
    from immuneML.encodings.MyKmerFrequencyEncoder import MyKmerFrequencyEncoder
    from immuneML.environment.Constants import Constants
    from immuneML.environment.EnvironmentSettings import EnvironmentSettings
    from immuneML.environment.LabelConfiguration import LabelConfiguration
    from immuneML.util.PathBuilder import PathBuilder
    from immuneML.util.RepertoireBuilder import RepertoireBuilder


    class TestKmerFrequencyEncoder(TestCase):

        def setUp(self) -> None: # useful if cache is used in the encoding (not used in this tutorial)
            os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

        def test_encode(self):
            path = EnvironmentSettings.tmp_test_path / "my_kmer_freq_enc/"

            PathBuilder.build(path)

            # create a dataset
            repertoires, metadata = RepertoireBuilder.build([["AAAT"], ["TAAA"]], path, {'l1': [1, 0]})
            dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata)

            # create an encoder
            encoder = MyKmerFrequencyEncoder.build_object(dataset, **{"k": 3, 'name': 'my_encoder'})

            lc = LabelConfiguration()
            lc.add_label("l1", [1, 2])

            encoded_dataset = encoder.encode(dataset, EncoderParams(
                result_path=path / "encoded_dataset",
                label_config=lc,
                learn_model=True,
                model={},
                filename="dataset.pkl"
            ))

            shutil.rmtree(path)

            # check if the output is as expected
            self.assertTrue(isinstance(encoded_dataset, RepertoireDataset))
            self.assertEqual(-1., np.round(encoded_dataset.encoded_data.examples[0, 2], 2))
            self.assertEqual(1., np.round(encoded_dataset.encoded_data.examples[0, 1], 2))
            self.assertTrue(isinstance(encoder, MyKmerFrequencyEncoder))



Adding an encoder: additional information
------------------------------------------

Encoders for different dataset types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inside immuneML, three different types of datasets are considered: :py:obj:`~immuneML.data_model.dataset.RepertoireDataset.RepertoireDataset` for immune
repertoires, :py:obj:`~immuneML.data_model.dataset.SequenceDataset.SequenceDataset` for single-chain immune
receptor sequences and :py:obj:`~immuneML.data_model.dataset.ReceptorDataset.ReceptorDataset` for paired sequences. We need to deal with encoding separately for each dataset type.

When an encoding only makes sense for one possible dataset type, for example RepertoireDataset, the new encoder class can simply inherit
DatasetEncoder and implement its abstract methods, and the :code:`build_object(dataset:Dataset, params)` method should return an instance of the class
itself when a correct dataset is given. An example of this is the encoder from this tutorial or :py:obj:`~immuneML.encodings.filtered_sequence_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder`.

Alternatively, when an encoding can be generalized for multiple dataset types, encoder classes are organized in the following manner:

    #. A base encoder class implements the method :code:`build_object(dataset: Dataset, params)`, that returns the correct dataset type-specific encoder. This encoder is a subclass of the base encoder class.

    #. The dataset type-specific subclasses implement all the abstract methods that differ between different dataset types.

    #. Each encoder class has to implement the function :code:`encode(dataset, params)` which returns a dataset object with encoded data parameter set.

It is not necessary to implement the encoding for all dataset types, since some encodings might not make sense for some dataset types. In that case,
if such a combination is specified (i.e., if the method :code:`build_object(dataset: Dataset, params)` receives an illegal dataset type), the encoder class
should raise an error with a user-friendly error message and the process will be terminated.

.. include:: ./dev_docs_util.rst

Implementing the encode() method in a new encoder class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The encode() method is called by immuneML to encode a new dataset. This method should be called with two arguments: a dataset and params
(an :py:obj:`~immuneML.encodings.EncoderParams.EncoderParams` object). The EncoderParams objects contains useful information such as the path where optional files with intermediate data can be
stored (such as vectorizer files, normalization, etc.), a :py:obj:`~immuneML.environment.LabelConfiguration.LabelConfiguration` object containing the
labels that were specified for the analysis, and more.

The :code:`encode()` method should return a new dataset object, which is a copy of the original input dataset, but with an added :code:`encoded_data` attribute.
The :code:`encoded_data` attribute should contain an :py:obj:`~immuneML.data_model.encoded_data.EncodedData.EncodedData` object, which is created with the
following arguments:

  - examples: a design matrix where the rows are repertoires, receptors or sequences, and the columns the encoding-specific features
  - encoding: a string denoting the encoder base class that was used.
  - labels: a dictionary of labels, where each label is a key, and the values are the label values across the examples (for example: {disease1: [positive, positive, negative]} if there are 3 repertoires)
  - example_ids: an optional list of identifiers for the examples (repertoires, receptors or sequences).
  - feature_names: an optional list of feature names, i.e., the names given to the encoding-specific features. When included, list must be as long as the number of features.
  - feature_annotations: an optional pandas dataframe with additional information about the features. When included, number of rows in this dataframe must correspond to the number of features.
  - info: an optional dictionary that may be used to store any additional information that is relevant (for example paths to additional output files).

The :code:`examples` attribute of the :code:`EncodedData` objects will be directly passed to the ML models for training. Other attributes are used for reports and
interpretability.

Unit testing
^^^^^^^^^^^^^^

To add a test for the new encoding, create a package under :code:`test.encodings` with the same name as the package created for adding the encoder class.
Implement a test method that ensures the encoder functions correctly for each relevant dataset type. A useful class here is
:py:obj:`~immuneML.simulation.dataset_generation.RandomDatasetGenerator.RandomDatasetGenerator`, which can create a dataset with random sequences.

Adding class documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Class documentation should be added as a docstring to the base encoder class. The documentation should include:

  #. A general description of how the data is encoded,
  #. A list of arguments with types and values,
  #. An example of how such an encoder should be defined in the YAML specification.

The class docstrings are used to automatically generate the documentation for the encoder. If an encoder should always be used in combination with a
specific report or ML method, it is possible to refer to these classes by name and create a link to the documentation of that class. For example,
the documentation of :py:obj:`~immuneML.encodings.reference_encoding.MatchedReceptorsEncoder.MatchedReceptorsEncoder` states ‘This encoding should be
used in combination with the :ref:`Matches` report’.

This is the example of documentation for :py:obj:`~immuneML.encodings.filtered_sequence_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder`:

.. code-block:: RST

  This encoder represents the repertoires as vectors where:
    - the first element corresponds to the number of label-associated clonotypes
    - the second element is the total number of unique clonotypes

    To determine what clonotypes (with features defined by comparison_attributes) are label-associated
    based on a statistical test. The statistical test used is Fisher's exact test (one-sided).

    Reference: Emerson, Ryan O. et al.
    ‘Immunosequencing Identifies Signatures of Cytomegalovirus Exposure History and HLA-Mediated Effects on the T Cell Repertoire’.
    Nature Genetics 49, no. 5 (May 2017): 659–65. `doi.org/10.1038/ng.3822 <https://doi.org/10.1038/ng.3822>`_.


    Arguments:

        comparison_attributes (list): The attributes to be considered to group receptors into clonotypes. Only the fields specified in
        comparison_attributes will be considered, all other fields are ignored. Valid comparison value can be any repertoire field name.

        p_value_threshold (float): The p value threshold to be used by the statistical test.

        sequence_batch_size (int): The number of sequences in a batch when comparing sequences across repertoires, typically 100s of thousands.
        This does not affect the results of the encoding, only the speed.

        repertoire_batch_size (int): How many repertoires will be loaded at once. This does not affect the result of the encoding, only the speed.
        This value is a trade-off between the number of repertoires that can fit the RAM at the time and loading time from disk.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_sa_encoding:
            SequenceAbundance:
                comparison_attributes:
                    - sequence_aas
                    - v_genes
                    - j_genes
                    - chains
                    - region_types
                p_value_threshold: 0.05
                sequence_batch_size: 100000
                repertoire_batch_size: 32
