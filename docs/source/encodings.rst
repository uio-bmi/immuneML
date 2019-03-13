###############
Encodings
###############

.. toctree::
   :maxdepth: 2

In order to enable applying machine learning to immune receptor data, it is necessary to represent the receptor data in a
manner suitable for machine learning algorithms. These transformations are implemented in the *encodings* package.

All of the encoders inherit DatasetEncoder class and provide the following methods:

.. code-block:: python


    def encode(dataset: Dataset, params: dict) -> Dataset:
        ...

    def store(encoded_dataset: Dataset, params: dict):
        ...

    def validate_configuration(params: dict):
        ...

All of them take in a *Dataset* object that they will encode. For more information on the dataset representation, see
:ref:`Dataset` and :ref:`Data model` sections.

In addition to the parameters already in the original dataset object, the returned dataset object contains an encoded_data
parameter of the following format:

.. code-block:: python

    {
        'repertoires': [...],   # mandatory field: encoded repertoires as a numpy matrix or a sparse matrix
        'labels': [...],        # mandatory field: list of labels per repertoire or a matrix of labels per repertoire
        'label_names': [..],    # optional field: list of label names
        'feature_names': [...]  # optional field: list of feature names
    }

Encoders based on k-mer frequencies
===================================

The **KmerFrequencyEncoder** class encodes a repertoire by frequencies of k-mers in all of the sequences of that repertoire.
The ``k`` parameter as well as what kind of k-mer should be checked is defined by the user. ``k`` is an integer, with
3 being a typical value. K-mer definition is given in the next section, :ref:`K-mer types and sequence encodings`.

K-mer types and sequence encodings
----------------------------------

K-mer is a sequence of letters of length k into which an immune receptor sequence can be decomposed.
For the purpose of immune receptor data representation, the following k-mer representations can be used:

*   k-mers as sequences of letters of length k (either overlapping or non-overlapping),
*   gapped k-mers as subsequences of letters of length k with gaps between them,
*   IMGT k-mers as sequences of letters of length k with included positional information according to the IMGT scheme,
*   IMGT gapped k-mers as subsequences of letters of length k with gaps between them and positional information
    according to the IMGT scheme.

Each of these k-mer representation corresponds to a class which implements an encoding of a sequence according to the
given strategy. *KmerFrequencyEncoder* can use any of these classes to encode a sequence (e.g. get a list of k-mers for
an immune receptor sequence) and then compute the frequencies in the repertoire.

Frequencies can be obtained in one of the following ways:

*   by calculating relative frequencies of k-mers in the repertoire or
*   by using L2 normalization.

To specify the encoding, one must define the following parameters:

.. code-block:: python

        {
            "normalization_type": NormalizationType.RELATIVE_FREQUENCY,         # relative frequencies of k-mers or L2
            "reads": ReadsType.UNIQUE,                                          # unique or all
            "sequence_encoding_strategy": SequenceEncodingType.CONTINUOUS_KMER, # continuous k-mers, gapped, IMGT-annotated or not
            "k": 3,                                                             # k-mer length
            ...
        }

Reads type signify whether the number of sequence occurrences in the repertoire will be taken into account. If ``unique``,
only unique sequences are encoded, and if ``all``, the number of sequence occurrences is incorporated into the frequencies
of k-mers created from the sequence.

Sequence encoding types can be:

1.  GAPPED_KMER
2.  CONTINUOUS_KMER
3.  IMGT_CONTINUOUS_KMER
4.  IMGT_GAPPPED_KMER
5.  IDENTITY

IMGT encodings include positional information of k-mers in the sequence. Identity sequence encoding returns the whole sequence and
measures the frequency of sequences in repertoires.

Encoding by vector representations
==================================

Another common representation for text analysis and biological sequence representation is based on vector representations.
One such representation is defined by Mikolov et al. [1]_ and is used here to infer vector representations of immune data.

Word2Vec encoding
-----------------

**Word2VecEncoder** implements the methods from *DatasetEncoder* and learns the vector representations of k-mers from the
context the k-mers appear in. To specify the encoding, one must define the following parameters:

.. code-block:: python

    {
        "model": {
            "k": 3,                                 # k-mer length
            "model_creator": ModelType.SEQUENCE,    # the context definition
            "size": 16                              # size of the vector to be learnt
        },
        ...
    }

Model creator in this setting defines the context which will be used to infer the representation of the sequence. Currently,
two types of contexts are supported in the ImmuneML:

*   ModelType.SEQUENCE - the context in which k-mer appears is the sequence it occurs in and similar k-mers are the ones
    from the same sequence (e.g. if the sequence is ``CASTTY`` and k-mer is ``AST``, then its context consists of k-mers
    ``CAS``, ``STT``, ``TTY``)
*   ModelType.KMER_PAIR - the context for the k-mer is defined as all the k-mers that are within one Hamming distance
    from the given k-mer (e.g. for k-mer ``CAS``, the context consists of ``CAA``, ``CAC``, ``CAD`` etc.).



.. [1] Efficient Estimation of Word Representations in Vector Space, Mikolov et al., arxiv, 2013, https://arxiv.org/abs/1301.3781