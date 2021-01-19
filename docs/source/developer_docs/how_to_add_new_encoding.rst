How to add a new encoding
===========================

To add a new encoder to immuneML, add a new package to source.encodings and create a new class there, which inherits
:py:obj:`source.encodings.DatasetEncoder.DatasetEncoder`. The name of the
class should be descriptive and user-friendly, and has to end with ‘Encoder’. The name prefix (everything before ‘Encoder’) will be the name that the
user should specify in the YAML specification (e.g., for an encoder called ‘SequenceAbundanceEncoder’, the name referenced in the specification will
be ‘SequenceAbundance’).

Encoders for different dataset types
-------------------------------------

Inside immuneML, three different types of datasets are considered: :py:obj:`source.data_model.dataset.RepertoireDataset.RepertoireDataset` for immune
repertoires, :py:obj:`source.data_model.dataset.SequenceDataset.SequenceDataset` for single-chain immune
receptor sequences and :py:obj:`source.data_model.dataset.ReceptorDataset.ReceptorDataset` for paired sequences. We need to deal with encoding separately for each dataset type.

When an encoding only makes sense for one possible dataset type, for example RepertoireDataset, the new encoder class can simply inherit
DatasetEncoder and implement its abstract methods, and the `build_object(dataset:Dataset, params)` method should return an instance of the class
itself when a correct dataset is given. An example of this is the :py:obj:`source.encodings.filtered_sequence_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder`.

Alternatively, when an encoding can be generalized for multiple dataset types, encoder classes are organized in the following manner:
  #. A base encoder class implements the method build_object(dataset: Dataset, params), which returns the correct dataset type-specific encoder.
  This encoder is a subclass of the base encoder class.
  #. The dataset type-specific subclasses implement all the abstract methods that differ between different dataset types.
  #. Each encoder class has to implement the function encode(dataset, params) which returns a dataset object with encoded data parameter set.

It is not necessary to implement the encoding for all dataset types, since some encodings might not make sense for some dataset types. In that case,
if such a combination is specified (i.e., if the method `build_object(dataset: Dataset, params)` receives an illegal dataset type), the encoder class
should raise an error and the process will be terminated.

Implementing the encode() method in a new encoder class
---------------------------------------------------------

The encode() method is called by immuneML to encode a new dataset. This method should be called with two arguments: a dataset and params
(an :py:obj:`source.encodings.EncoderParams.EncoderParams` object). The EncoderParams objects contains useful information such as the path where optional files with intermediate data can be
stored (such as vectorizer files, normalization, etc.), a :py:obj:`source.environment.LabelConfiguration.LabelConfiguration` object containing the
labels that were specified for the analysis, and more.

The `encode()` method should return a new dataset object, which is a copy of the original input dataset, but with an added `encoded_data` attribute.
The `encoded_data` attribute should contain an :py:obj:`source.data_model.encoded_data.EncodedData.EncodedData` object, which is created with the
following arguments:

  - examples: a design matrix where the rows are repertoires, receptors or sequences, and the columns the encoding-specific features
  - encoding: a string denoting the encoder base class that was used.
  - labels: a dictionary of labels, where each label is a key, and the values are the label values across the examples (for example: {disease1:
  [positive, positive, negative]} if there are 3 repertoires)
  - example_ids: an optional list of identifiers for the examples (repertoires, receptors or sequences).
  - feature_names: an optional list of feature names, i.e., the names given to the encoding-specific features. When included, list must be as long as
  the number of features.
  - feature_annotations: an optional pandas dataframe with additional information about the features. When included, number of rows in this dataframe
  must correspond to the number of features.
  - info: an optional dictionary that may be used to store any additional information that is relevant (for example paths to additional output files).

The `examples` attribute of the `EncodedData` objects will be directly passed to the ML models for training. Other attributes are used for reports and
interpretability.

Unit testing
-------------

To add a test for the new encoding, create a package under `test.encodings` with the same name as the package created for adding the encoder class.
Implement a test method that ensures the encoder functions correctly for each relevant dataset type. A useful class here is
:py:obj:`source.simulation.dataset_generation.RandomDatasetGenerator.RandomDatasetGenerator`, which can create a dataset with random sequences.

Adding class documentation
---------------------------

Class documentation should be added as a docstring to the base encoder class. The documentation should include:

  #. A general description of how the data is encoded,
  #. A list of arguments with types and values,
  #. An example of how such an encoder should be defined in the YAML specification.

The class docstrings are used to automatically generate the documentation for the encoder. If an encoder should always be used in combination with a
specific report or ML method, it is possible to refer to these classes by name and create a link to the documentation of that class. For example,
the documentation of :py:obj:`source.encodings.reference_encoding.MatchedReceptorsEncoder.MatchedReceptorsEncoder` states ‘This encoding should be
used in combination with the :ref:`Matches` report’.

This is the example of documentation for :py:obj:`source.encodings.filtered_sequence_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder`:

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