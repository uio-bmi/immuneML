
CompAIRRDistance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Encodes a given RepertoireDataset as a distance matrix, using the Morisita-Horn distance metric.
Internally, `CompAIRR <https://github.com/uio-bmi/compairr/>`_ is used for fast calculation of overlap between repertoires.
This creates a pairwise distance matrix between each of the repertoires.
The distance is calculated based on the number of matching receptor chain sequences between the repertoires. This matching may be
defined to permit 1 or 2 mismatching amino acid/nucleotide positions and 1 indel in the sequence. Furthermore,
matching may or may not include V and J gene information, and sequence frequencies may be included or ignored.

When mismatches (differences and indels) are allowed, the Morisita-Horn similarity may exceed 1. In this case, the
Morisita-Horn distance (= similarity - 1) is set to 0 to avoid negative distance scores.


**Dataset type:**

- RepertoireDatasets


**Specification arguments:**

- compairr_path (Path): optional path to the CompAIRR executable. If not given, it is assumed that CompAIRR has been
  installed such that it can be called directly on the command line with the command 'compairr', or that it is
  located at /usr/local/bin/compairr.

- keep_compairr_input (bool): whether to keep the input file that was passed to CompAIRR. This may take a lot of
  storage space if the input dataset is large. By default, the input file is not kept.

- differences (int): Number of differences allowed between the sequences of two immune receptor chains, this may be
  between 0 and 2. By default, differences is 0.

- indels (bool): Whether to allow an indel. This is only possible if differences is 1. By default, indels is False.

- ignore_counts (bool): Whether to ignore the frequencies of the immune receptor chains. If False, frequencies will
  be included, meaning the 'counts' values for the receptors available in two repertoires are multiplied. If False,
  only the number of unique overlapping immune receptors ('clones') are considered. By default, ignore_counts is False.

- ignore_genes (bool): Whether to ignore V and J gene information. If False, the V and J genes between two receptor
  chains have to match. If True, gene information is ignored. By default, ignore_genes is False.

- threads (int): The number of threads to use for parallelization. Default is 8.


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        encodings:
            my_distance_encoder:
                CompAIRRDistance:
                    compairr_path: optional/path/to/compairr
                    differences: 0
                    indels: False
                    ignore_counts: False
                    ignore_genes: False



CompAIRRSequenceAbundance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


This encoder works similarly to the :py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder`,
but internally uses `CompAIRR <https://github.com/uio-bmi/compairr/>`_ to accelerate core computations.

This encoder represents the repertoires as vectors where:

- the first element corresponds to the number of label-associated clonotypes
- the second element is the total number of unique clonotypes

To determine what clonotypes (amino acid sequences with or without matching V/J genes) are label-associated, Fisher's exact test (one-sided)
is used.

The encoder also writes out files containing the contingency table used for fisher's exact test,
the resulting p-values, and the significantly abundant sequences
(use :py:obj:`~immuneML.reports.encoding_reports.RelevantSequenceExporter.RelevantSequenceExporter` to export these sequences in AIRR format).

Reference: Emerson, Ryan O. et al.
‘Immunosequencing Identifies Signatures of Cytomegalovirus Exposure History and HLA-Mediated Effects on the T Cell Repertoire’.
Nature Genetics 49, no. 5 (May 2017): 659–65. `doi.org/10.1038/ng.3822 <https://doi.org/10.1038/ng.3822>`_.

Note: to use this encoder, it is necessary to explicitly define the positive class for the label when defining the label
in the instruction. With positive class defined, it can then be determined which sequences are indicative of the positive class.
See :ref:`Reproduction of the CMV status predictions study` for an example using :py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder`.

**Dataset type:**

- RepertoireDatasets


**Specification arguments:**

- p_value_threshold (float): The p value threshold to be used by the statistical test.

- compairr_path (Path): optional path to the CompAIRR executable. If not given, it is assumed that CompAIRR
  has been installed such that it can be called directly on the command line with the command 'compairr',
  or that it is located at /usr/local/bin/compairr.

- ignore_genes (bool): Whether to ignore V and J gene information. If False, the V and J genes between two receptor chains
  have to match. If True, gene information is ignored. By default, ignore_genes is False.

- sequence_batch_size (int): The number of sequences in a batch when comparing sequences across repertoires, typically 100s of thousands.
  This does not affect the results of the encoding, but may affect the speed and memory usage. The default value is 1.000.000

- threads (int): The number of threads to use for parallelization. This does not affect the results of the encoding, only the speed.
  The default number of threads is 8.

- keep_temporary_files (bool): whether to keep temporary files, including CompAIRR input, output and log files, and the sequence
  presence matrix. This may take a lot of storage space if the input dataset is large. By default, temporary files are not kept.


**YAML specification:**

.. code-block:: yaml

    definitions:
        encodings:
            my_sa_encoding:
                CompAIRRSequenceAbundance:
                    compairr_path: optional/path/to/compairr
                    p_value_threshold: 0.05
                    ignore_genes: False
                    threads: 8



Composite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


This encoder allows to combine multiple different encodings together, for example, KmerFrequency encoder
with VGeneEncoder. The parameters for the different encoders are passed as a list of dictionaries, where each
dictionary contains the parameters for one encoder. The different encoders are applied sequentially and their
results concatenated together.

**Dataset type:**
- SequenceDatasets
- ReceptorDatasets
- RepertoireDatasets

.. note::

    To combine multiple encodings (e.g., GeneFrequency and KmerFrequency), keep in mind how the ML method will
    use the encoded data downstream. Currently, the recommended way to use CompositeEncoder is with
    :ref:`LogRegressionCustomPenalty`, where you can specify which features should not be penalized.

**Specification arguments:**

- encoders (list): A list of dictionaries, where each dictionary contains the parameters for one encoder.

**YAML specification:**

.. code-block:: yaml

    encodings:
        my_composite_encoding:
            Composite:
                encoders:
                    - KmerFrequency:
                        k: 3
                    - GeneFrequency:
                        genes: [V]
                        normalization_type: relative_frequency
                        scale_to_unit_variance: true
                        scale_to_zero_mean: true



DeepRC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


DeepRCEncoder should be used in combination with the DeepRC ML method (:ref:`DeepRC`).
This encoder writes the data in a RepertoireDataset to .tsv files.
For each repertoire, one .tsv file is created containing the amino acid sequences and the counts.
Additionally, one metadata .tsv file is created, which describes the subset of repertoires that is encoded by
a given instance of the DeepRCEncoder.

Note: sequences where count is None, the count value will be set to 1

**Dataset type:**

- RepertoireDatasets


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        encodings:
            my_deeprc_encoder: DeepRC



Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Encodes a given RepertoireDataset as distance matrix, where the pairwise distance between each of the repertoires
is calculated. The distance is calculated based on the presence/absence of elements defined under attributes_to_match.
Thus, if attributes_to_match contains only 'sequence_aas', this means the distance between two repertoires is maximal
if they contain the same set of sequence_aas, and the distance is minimal if none of the sequence_aas are shared between
two repertoires.

**Specification arguments:**

- distance_metric (:py:mod:`~immuneML.encodings.distance_encoding.DistanceMetricType`): The metric used to calculate the
  distance between two repertoires. Valid values are: `JACCARD`, `MORISITA_HORN`.
  The default distance metric is JACCARD (inverse Jaccard).

- sequence_batch_size (int): The number of sequences to be processed at once. Increasing this number increases the memory use.
  The default value is 1000.

- attributes_to_match (list): The attributes to consider when determining whether a sequence is present in both repertoires.
  Only the fields defined under attributes_to_match will be considered, all other fields are ignored.
  Valid values include any repertoire attribute as defined in AIRR rearrangement schema (cdr3_aa, v_call, j_call, etc).

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        encodings:
            my_distance_encoder:
                Distance:
                    distance_metric: JACCARD
                    sequence_batch_size: 1000
                    attributes_to_match:
                        - cdr3_aa
                        - v_call
                        - j_call



ESMC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Encoder based on a pretrained protein language model by Hayes et al. 2025. The used transformer model is
"esmc_300m".

Original publication:
Hayes, T., Rao, R., Akin, H., Sofroniew, N. J., Oktay, D., Lin, Z., Verkuil, R., Tran, V. Q., Deaton, J.,
Wiggert, M., Badkundri, R., Shafkat, I., Gong, J., Derry, A., Molina, R. S., Thomas, N., Khan, Y. A.,
Mishra, C., Kim, C., … Rives, A. (2025). Simulating 500 million years of evolution with a language model.
Science, 387(6736), 850–858. https://doi.org/10.1126/science.ads0018

Original GitHub repository with license information: https://github.com/evolutionaryscale/esm

**Dataset type:**

- SequenceDatasets

- ReceptorDatasets

- RepertoireDatasets

**Specification arguments:**

- region_type (RegionType): Which part of the receptor sequence to encode. Defaults to IMGT_CDR3.

- device (str): Which device to use for model inference - 'cpu', 'cuda', 'mps' - as defined by pytorch.
  Defaults to 'cpu'.

- num_processes (int): Number of processes to use for parallel processing. Defaults to 1.

- batch_size (int): The number of sequences to encode at the same time. This could have large impact on memory usage.
  If memory is an issue, try with smaller batch sizes. Defaults to 4096.

- scale_to_zero_mean (bool): Whether to scale the embeddings to zero mean. Defaults to True.

- scale_to_unit_variance (bool): Whether to scale the embeddings to unit variance. Defaults to True.

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        encodings:
            my_emsc_encoder:
                ESMC:
                    region_type: IMGT_CDR3
                    device: cpu
                    num_processes: 4
                    batch_size: 4096



EvennessProfile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


The EvennessProfileEncoder class encodes a repertoire based on the clonal frequency distribution. The evenness for
a given repertoire is defined as follows:

.. math::

    ^{\alpha} \mathrm{E}(\mathrm{f})=\frac{\left(\sum_{\mathrm{i}=1}^{\mathrm{n}} \mathrm{f}_{\mathrm{i}}^{\alpha}\right)^{\frac{1}{1-\alpha}}}{\mathrm{n}}

That is, it is the exponential of Renyi entropy at a given alpha divided by the species richness, or number of unique
sequences.

Reference: Greiff et al. (2015). A bioinformatic framework for immune repertoire diversity profiling enables detection of immunological
status. Genome Medicine, 7(1), 49. `doi.org/10.1186/s13073-015-0169-8 <https://doi.org/10.1186/s13073-015-0169-8>`_

**Dataset type:**

- RepertoireDatasets


**Specification arguments:**

- min_alpha (float): minimum alpha value to use

- max_alpha (float): maximum alpha value to use

- dimension (int): dimension of output evenness profile vector, or the number of alpha values to linearly space
  between min_alpha and max_alpha

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        encodings:
            my_evenness_profile:
                EvennessProfile:
                    min_alpha: 0
                    max_alpha: 10
                    dimension: 51




GeneFrequency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


GeneFrequencyEncoder represents a repertoire by the frequency of V and/or J genes used.

**Dataset type:**
- RepertoireDatasets

**Specification arguments:**

- genes (list): List of genes to use for the encoding. Possible values are 'V', and 'J'. At least one gene must be
  specified.

- normalization_type (str): Type of normalization to apply to the gene frequencies. Possible values are 'none',
  'binary', 'relative_frequency', 'max', 'l2'. Defaults to 'relative_frequency'.

- scale_to_zero_mean (bool): Whether to scale the features to zero mean. Defaults to True.

- scale_to_unit_variance (bool): Whether to scale the features to unit variance. Defaults to True.

**YAML specification:**

.. code-block:: yaml

    encodings:
        gene_frequency_encoding:
            GeneFrequency:
                genes: [V, J]
                normalization_type: relative_frequency
                scale_to_unit_variance: true
                scale_to_zero_mean: true



KmerAbundance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


This encoder is related to the :py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder`,
but identifies label-associated subsequences (k-mers) instead of full label-associated sequences.

This encoder represents the repertoires as vectors where:

- the first element corresponds to the number of label-associated k-mers found in a repertoire
- the second element is the total number of unique k-mers per repertoire

The label-associated k-mers are determined based on a one-sided Fisher's exact test.

The encoder also writes out files containing the contingency table used for fisher's exact test,
the resulting p-values, and the significantly abundant k-mers.

Note: to use this encoder, it is necessary to explicitly define the positive class for the label when defining the label
in the instruction. With positive class defined, it can then be determined which sequences are indicative of the positive class.
See :ref:`Reproduction of the CMV status predictions study` for an example using :py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder`.

**Dataset type:**

- RepertoireDatasets


**Specification arguments:**

- p_value_threshold (float): The p value threshold to be used by the statistical test.

- sequence_encoding (:py:mod:`~immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType`): The type of k-mers that are used. The simplest (default) sequence_encoding is :py:mod:`~immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.CONTINUOUS_KMER`, which uses contiguous subsequences of length k to represent the k-mers. When gapped k-mers are used (:py:mod:`~immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.GAPPED_KMER`, :py:mod:`~immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.GAPPED_KMER`), the k-mers may contain gaps with a size between min_gap and max_gap, and the k-mer length is defined as a combination of k_left and k_right. When IMGT k-mers are used (:py:mod:`~immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.IMGT_CONTINUOUS_KMER`, :py:mod:`~immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.IMGT_GAPPED_KMER`), IMGT positional information is taken into account (i.e. the same sequence in a different position is considered to be a different k-mer).

- k (int): Length of the k-mer (number of amino acids) when ungapped k-mers are used. The default value for k is 3.

- k_left (int): When gapped k-mers are used, k_left indicates the length of the k-mer left of the gap. The default value for k_left is 1.

- k_right (int): Same as k_left, but k_right determines the length of the k-mer right of the gap. The default value for k_right is 1.

- min_gap (int): Minimum gap size when gapped k-mers are used. The default value for min_gap is 0.

- max_gap: (int): Maximum gap size when gapped k-mers are used. The default value for max_gap is 0.


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        encodings:
            my_ka_encoding:
                KmerAbundance:
                    p_value_threshold: 0.05
                    threads: 8



KmerFrequency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


The KmerFrequencyEncoder class encodes a repertoire, sequence or receptor by frequencies of k-mers it contains.
A k-mer is a sequence of letters of length k into which an immune receptor sequence can be decomposed.
K-mers can be defined in different ways, as determined by the sequence_encoding. If a dataset contains receptor
sequences from multiple loci (e.g., TRA and TRB), the k-mer frequencies will be computed per locus and then combined
into a single feature vector per sample. The k-mer frequencies can be normalized in different ways, as determined by
the normalization_type. The design matrix can optionally be scaled to unit variance and/or to zero mean. The k-mer
frequencies can be computed based on unique sequences (clonotypes) or taking into account the counts of the
sequences in the repertoire.

**Dataset type:**

- SequenceDatasets

- ReceptorDatasets

- RepertoireDatasets


**Specification arguments:**

- sequence_encoding (:py:mod:`~immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType`):
  Sequence encoding determines how the sequences are decomposed into k-mers. It includes:

  - CONTINUOUS_KMER: contiguous overlapping k-mers of length k (e.g., ACDE -> {ACD, CDE} for k=3) - default value
  - GAPPED_KMER: k-mers of length k_left + k_right with a gap of size between min_gap and max_gap in between
    (e.g., ACDE -> {AC, A.D, CD, C.E} for k_left=1, k_right=1, min_gap=0, max_gap=1)
  - IMGT_CONTINUOUS_KMER: contiguous k-mers of length k with IMGT positional information
    (e.g., AHCDE -> {'AHC_105', 'HCD_106', 'CDE_107'} for k=3)
  - IMGT_GAPPED_KMER: k-mers of length k_left + k_right with a gap of size between min_gap and max_gap in
    between, annotated by the starting IMGT position (e.g., AHCDE -> {'AH_105', 'HC_106', 'CD_107', 'DE_116',
    'A.C_105', 'H.D_106', 'C.E_107'} for k_left=1, k_right=1, min_gap=0, max_gap=1)
  - V_GENE_CONT_KMER: contiguous k-mers of length k, annotated by the V gene of the sequence they belong to
    (e.g., ACDE -> {V1-1_ACD, V1-1_CDE} for k=3 and V gene V1-1)
  - V_GENE_IMGT_KMER: contiguous k-mers of length k, annotated by the V gene of the sequence they belong to,
    annotated by the starting IMGT position (e.g., AHCDE -> {V1-1_AHC_105, V1-1_HCD_106, V1-1_CDE_107} for k=3 and
    V gene V1-1)
  - IDENTITY: the k-mers correspond to the original sequences

- normalization_type (:py:mod:`~immuneML.analysis.data_manipulation.NormalizationType`): The way in which the
  k-mer frequencies should be normalized to unit norm; options are: binary, relative_frequency (also known as l1,
  default value), l2, max, none. For more information, see scikit-learn's documentation on
  `normalization <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html#sklearn.preprocessing.normalize>`_.

- reads (:py:mod:`~immuneML.util.ReadsType`): Reads type signify whether the counts of the sequences in the
  repertoire will be taken into account. If :py:mod:`~immuneML.util.ReadsType.UNIQUE`, only unique sequences
  (clonotypes) are encoded, and if :py:mod:`~immuneML.util.ReadsType.ALL`, the sequence 'count' value is taken into
  account when determining the k-mer frequency. The default value for reads is unique.

- k (int): Length of the k-mer (number of amino acids) when ungapped k-mers are used. The default value for k is 3.

- k_left (int): When gapped k-mers are used, k_left indicates the length of the k-mer left of the gap. The default
  value for k_left is 1.

- k_right (int): Same as k_left, but k_right determines the length of the k-mer right of the gap. The default value
  for k_right is 1.

- min_gap (int): Minimum gap size when gapped k-mers are used. The default value for min_gap is 0.

- max_gap: (int): Maximum gap size when gapped k-mers are used. The default value for max_gap is 0.

- sequence_type (str): Whether to work with nucleotide or amino acid sequences. Amino acid sequences are the
  default. To work with either sequence type, the sequences of the desired type should be included in the datasets,
  e.g., listed under 'columns_to_load' parameter. By default, both types will be included if available. Valid values
  are: AMINO_ACID and NUCLEOTIDE.

- scale_to_unit_variance (bool): whether to scale the design matrix after normalization to have unit variance per
  feature. Setting this argument to True might improve the subsequent classifier's performance depending on the type
  of the classifier. The default value for scale_to_unit_variance is true.

- scale_to_zero_mean (bool): whether to scale the design matrix after normalization to have zero mean per feature.
  Setting this argument to True might improve the subsequent classifier's performance depending on the type of the
  classifier. However, if the original design matrix was sparse, setting this argument to True will destroy the
  sparsity and will increase the memory consumption. The default value for scale_to_zero_mean is false.

- region_type (:py:mod:`~immuneML.data_model.SequenceParams.RegionType): the part of the receptor sequence to use
  in the analysis. The default value is IMGT_CDR3. Other values: IMGT_CDR1, IMGT_CDR2, IMGT_CDR3, IMGT_FR1,
  IMGT_FR2, IMGT_FR3, IMGT_FR4, IMGT_JUNCTION, FULL_SEQUENCE. Note that if an IMGT-based sequence encoding is
  used, the region_type has to be IMGT_CDR3 or IMGT_JUNCTION.


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        encodings:
            my_continuous_kmer:
                KmerFrequency:
                    normalization_type: RELATIVE_FREQUENCY
                    reads: UNIQUE
                    sequence_encoding: CONTINUOUS_KMER
                    sequence_type: NUCLEOTIDE
                    k: 3
                    scale_to_unit_variance: True
                    scale_to_zero_mean: True
            my_gapped_kmer:
                KmerFrequency:
                    normalization_type: RELATIVE_FREQUENCY
                    reads: UNIQUE
                    sequence_encoding: GAPPED_KMER
                    sequence_type: AMINO_ACID
                    k_left: 2
                    k_right: 2
                    min_gap: 1
                    max_gap: 3
                    scale_to_unit_variance: True
                    scale_to_zero_mean: False



MatchedReceptors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Encodes the dataset based on the matches between a dataset containing unpaired (single chain) data,
and a paired reference receptor dataset.
For each paired reference receptor, the frequency of either chain in the dataset is counted.

This encoding can be used in combination with the :ref:`Matches` report.

When sum_matches and normalize are set to True, this encoder behaves similarly as described in: Yao, Y. et al. ‘T cell receptor repertoire as a potential diagnostic marker for celiac disease’.
Clinical Immunology Volume 222 (January 2021): 108621. `doi.org/10.1016/j.clim.2020.108621 <https://doi.org/10.1016/j.clim.2020.108621>`_
with the only exception being that this encoder uses paired receptors, while the original publication used single sequences (see also: :ref:`MatchedSequences` encoder).


**Dataset type:**

- RepertoireDatasets


**Specification arguments:**

- reference (dict): A dictionary describing the reference dataset file. Import should be specified the same way as
  regular dataset import. It is only allowed to import a receptor dataset here (i.e., is_repertoire is False and
  paired is True by default, and these are not allowed to be changed).

- max_edit_distances (dict): A dictionary specifying the maximum edit distance between a target sequence (from the
  repertoire) and the reference sequence. A maximum distance can be specified per chain, for example to allow for
  less strict matching of TCR alpha and BCR light chains. When only an integer is specified, this distance is
  applied to all possible chains.

- reads (:py:mod:`~immuneML.util.ReadsType`): Reads type signify whether the counts of the sequences in the
  repertoire will be taken into account. If :py:mod:`~immuneML.util.ReadsType.UNIQUE`, only unique sequences
  (clonotypes) are counted, and if :py:mod:`~immuneML.util.ReadsType.ALL`, the sequence 'count' value is summed when
  determining the number of matches. The default value for reads is all.

- sum_matches (bool): When sum_matches is False, the resulting encoded data matrix contains multiple columns with
  the number of matches per reference receptor chain. When sum_matches is true, the columns representing each of the
  two chains are summed together, meaning that there are only two aggregated sums of matches (one per chain) per
  repertoire in the encoded data. To use this encoder in combination with the :ref:`Matches` report, sum_matches
  must be set to False. When sum_matches is set to True, this encoder behaves similarly to the encoder described by
  Yao, Y. et al. By default, sum_matches is False.

- normalize (bool): If True, the chain matches are divided by the total number of unique receptors in the repertoire
  (when reads = unique) or the total number of reads in the repertoire (when reads = all).


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        encodings:
            my_mr_encoding:
                MatchedReceptors:
                    reference:
                        format: VDJdb
                        params:
                            path: path/to/file.txt
                    max_edit_distances:
                        TRA: 1
                        TRB: 0


MatchedRegex
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Encodes the dataset based on the matches between a RepertoireDataset and a collection of regular expressions.
For each regular expression, the number of sequences in the RepertoireDataset containing the expression is counted.
This can also be used to count how often a subsequence occurs in a RepertoireDataset.

The regular expressions are defined per chain, and it is possible to require a V gene match in addition to the
CDR3 sequence containing the regular expression.

This encoding can be used in combination with the :ref:`Matches` report.


**Dataset type:**

- RepertoireDatasets


**Specification arguments:**

- match_v_genes (bool): Whether V gene matches are required. If this is True, a match is only counted if the
  V gene matches the gene specified in the motif input file. By default match_v_genes is False.

- reads (:py:mod:`~immuneML.util.ReadsType`): Reads type signify whether the counts of the sequences in the
  repertoire will be taken into account. If :py:mod:`~immuneML.util.ReadsType.UNIQUE`, only unique sequences
  (clonotypes) are counted, and if :py:mod:`~immuneML.util.ReadsType.ALL`, the sequence 'count' value is
  summed when determining the number of matches. The default value for reads is all.

- motif_filepath (str): The path to the motif input file. This should be a tab separated file containing a
  column named 'id' and for every chain that should be matched a column containing the regex (<chain>_regex) and a
  column containing the V gene (<chain>V) if match_v_genes is True.
  The chains are specified by their three-letter code, see :py:obj:`~immuneML.data_model.receptor.receptor_sequence.Chain.Chain`.

In the simplest case, when counting the number of occurrences of a given list of k-mers in TRB sequences, the
contents of the motif file could look like this:

====  ==========
id    TRB_regex
====  ==========
1     ACG
2     EDNA
3     DFWG
====  ==========

It is also possible to test whether paired regular expressions occur in the dataset (for example: regular expressions
matching both a TRA chain and a TRB chain) by specifying them on the same line.
In a more complex case where both paired and unpaired regular expressions are specified, in addition to matching the V
genes, the contents of the motif file could look like this:

====  ==========  =======  ==========  ========
id    TRA_regex   TRAV     TRB_regex   TRBV
====  ==========  =======  ==========  ========
1     AGQ.GSS     TRAV35   S[APL]GQY   TRBV29-1
2                          ASS.R.*     TRBV7-3
====  ==========  =======  ==========  ========


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        encodings:
            my_mr_encoding:
                MatchedRegex:
                    motif_filepath: path/to/file.txt
                    match_v_genes: True
                    reads: unique



MatchedSequences
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Encodes the dataset based on the matches between a RepertoireDataset and a reference sequence dataset. The feature
names are derived from the reference sequences: "v_call_sequence_j_call" (e.g., "TRBV12-3_CASSLGTDTQYF_TRBJ2-7"). If
there are duplicates in the feature names but sequences have different sequence IDs, the sequence ID is appended to
the feature name to make it unique.

This encoding can be used in combination with the :ref:`Matches` report.

When sum_matches and normalize are set to True, this encoder behaves as described in: Yao, Y. et al. ‘T cell receptor repertoire as a potential diagnostic marker for celiac disease’.
Clinical Immunology Volume 222 (January 2021): 108621. `doi.org/10.1016/j.clim.2020.108621 <https://doi.org/10.1016/j.clim.2020.108621>`_


**Dataset type:**

- RepertoireDatasets


**Specification arguments:**

- reference (dict): A dictionary describing the reference dataset file. Import should be specified the same way as
  regular dataset import. It is only allowed to import a sequence dataset here (i.e., is_repertoire and paired are
  False by default, and are not allowed to be set to True).

- max_edit_distance (int): The maximum edit distance between a target sequence (from the repertoire) and the
  reference sequence.

- reads (:py:mod:`~immuneML.util.ReadsType`): Reads type signify whether the counts of the sequences in the
  repertoire will be taken into account. If :py:mod:`~immuneML.util.ReadsType.UNIQUE`, only unique sequences
  (clonotypes) are counted, and if :py:mod:`~immuneML.util.ReadsType.ALL`, the sequence 'count' value is summed when
  determining the number of matches. The default value for reads is all.

- sum_matches (bool): When sum_matches is False, the resulting encoded data matrix contains multiple columns with
  the number of matches per reference sequence. When sum_matches is true, all columns are summed together, meaning
  that there is only one aggregated sum of matches per repertoire in the encoded data.
  To use this encoder in combination with the :ref:`Matches` report, sum_matches must be set to False. When
  sum_matches is set to True, this encoder behaves as described by Yao, Y. et al. By default, sum_matches is False.

- normalize (bool): If True, the sequence matches are divided by the total number of unique sequences in the
  repertoire (when reads = unique) or the total number of reads in the repertoire (when reads = all).

- output_count_as_feature: if True, the encoded repertoire is represented by the matches, and by the total number
  of sequences (or reads) in the repertoire, as defined by reads parameter above; by default this is False


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        encodings:
            my_ms_encoding:
                MatchedSequences:
                    reference:
                        format: VDJDB
                        params:
                            path: path/to/file.txt
                    max_edit_distance: 1



Metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Encoder that uses metadata fields as features, such as HLA.

**Dataset type:**
- RepertoireDatasets
- SequenceDatasets
- ReceptorDatasets

**Specification arguments:**

- metadata_fields (list): List of metadata fields to use as features.

**YAML specification:**

.. code-block:: yaml

    encodings:
        metadata_encoding:
            Metadata:
                metadata_fields: [HLA, sex]



Motif
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


This encoder enumerates every possible positional motif in a sequence dataset, and keeps only the motifs associated with the positive class.
A 'motif' is defined as a combination of position-specific amino acids. These motifs may contain one or multiple gaps.
Motifs are filtered out based on a minimal precision and recall threshold for predicting the positive class.

Note: the MotifEncoder can only be used for sequences of the same length.

The ideal recall threshold(s) given a user-defined precision threshold can be calibrated using the
:py:obj:`~immuneML.reports.data_reports.MotifGeneralizationAnalysis` report. It is recommended to first run this report
in :py:obj:`~immuneML.workflows.instructions.exploratory_analysis.ExploratoryAnalysisInstruction` before using this encoder for ML.

This encoder can be used in combination with the :py:obj:`~immuneML.ml_methods.BinaryFeatureClassifier` in order to
learn a minimal set of compatible motifs for predicting the positive class.
Alternatively, it may be combined with scikit-learn methods, such as for example :py:obj:`~immuneML.ml_methods.LogisticRegression`,
to learn a weight per motif.

**Dataset type:**

- SequenceDatasets


**Specification arguments:**

- max_positions (int): The maximum motif size. This is number of positional amino acids the motif consists of (excluding gaps). The default value for max_positions is 4.

- min_positions (int): The minimum motif size (see also: max_positions). The default value for max_positions is 1.

- no_gaps (bool): Must be set to True if only contiguous motifs (position-specific k-mers) are allowed. By default, no_gaps is False, meaning both gapped and ungapped motifs are searched for.

- min_precision (float): The minimum precision threshold for keeping a motif. The default value for min_precision is 0.8.

- min_recall (float): The minimum recall threshold for keeping a motif. The default value for min_precision is 0.
  It is also possible to specify a recall threshold for each motif size. In this case, a dictionary must be specified where
  the motif sizes are keys and the recall values are values. Use the :py:obj:`~immuneML.reports.data_reports.MotifGeneralizationAnalysis` report
  to calibrate the optimal recall threshold given a user-defined precision threshold to ensure generalisability to unseen data.

- min_true_positives (int): The minimum number of true positive sequences that a motif needs to occur in. The default value for min_true_positives is 10.

- candidate_motif_filepath (str): Optional filepath for pre-filterd candidate motifs. This may be used to save time. Only the given candidate motifs are considered.
  When this encoder has been run previously, a candidate motifs file named 'all_candidate_motifs.tsv' will have been exported. This file contains all
  possible motifs with high enough min_true_positives without applying precision and recall thresholds.
  The file must be a tab-separated file, structured as follows:

  ========  ==============
  indices    amino_acids
  ========  ==============
  1&2&3      A&G&C
  5&7        E&D
  ========  ==============

  The example above contains two motifs: AGC in positions 123, and E-D in positions 5-7 (with a gap at position 6).

- label (str): The name of the binary label to train the encoder for. This is only necessary when the dataset contains multiple labels.


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        encodings:
            my_motif_encoder:
                MotifEncoder:
                    max_positions: 4
                    min_precision: 0.8
                    min_recall:  # different recall thresholds for each motif size
                        1: 0.5   # For shorter motifs, a stricter recall threshold is used
                        2: 0.1
                        3: 0.01
                        4: 0.001
                    min_true_positives: 10



OneHot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


One-hot encoding for repertoires, sequences or receptors. In one-hot encoding, each alphabet character
(amino acid or nucleotide) is replaced by a sparse vector with one 1 and the rest zeroes. The position of the
1 represents the alphabet character.

**Dataset type:**

- SequenceDatasets

- ReceptorDatasets

- RepertoireDatasets


**Specification arguments:**

- use_positional_info (bool): whether to include features representing the positional information.
  If True, three additional feature vectors will be added, representing the sequence start, sequence middle
  and sequence end. The values in these features are scaled between 0 and 1. A graphical representation of
  the values of these vectors is given below.

.. code-block:: console

      Value of sequence start:         Value of sequence middle:        Value of sequence end:

    1 \                              1    /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\         1                          /
       \                                 /                   \                                  /
        \                               /                     \                                /
    0    \_____________________      0 /                       \      0  _____________________/
      <----sequence length---->        <----sequence length---->         <----sequence length---->


- distance_to_seq_middle (int): only applies when use_positional_info is True. This is the distance from the edge
  of the CDR3 sequence (IMGT positions 105 and 117) to the portion of the sequence that is considered 'middle'.
  For example: if distance_to_seq_middle is 6 (default), all IMGT positions in the interval [111, 112)
  receive positional value 1.
  When using nucleotide sequences: note that the distance is measured in (amino acid) IMGT positions.
  If the complete sequence length is smaller than 2 * distance_to_seq_middle, the maximum value of the
  'start' and 'end' vectors will not reach 0, and the maximum value of the 'middle' vector will not reach 1.
  A graphical representation of the positional vectors with a too short sequence is given below:


.. code-block:: console

    Value of sequence start         Value of sequence middle        Value of sequence end:
    with very short sequence:       with very short sequence:       with very short sequence:

         1 \                               1                                 1    /
            \                                                                    /
             \                                /\                                /
         0                                 0 /  \                            0
           <->                               <-->                               <->

- flatten (bool): whether to flatten the final onehot matrix to a 2-dimensional matrix [examples, other_dims_combined]
  This must be set to True when using onehot encoding in combination with scikit-learn ML methods (inheriting :py:obj:`~source.ml_methods.SklearnMethod.SklearnMethod`),
  such as :ref:`LogisticRegression`, :ref:`SVM`, :ref:`SVC`, :ref:`RandomForestClassifier` and :ref:`KNN`.

- sequence_type: whether to use nucleotide or amino acid sequence for encoding. Valid values are 'nucleotide' and 'amino_acid'.

- region_type: which part of the sequence to encode; e.g., imgt_cdr3, imgt_junction


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        encodings:
            one_hot_vanilla:
                OneHot:
                    use_positional_info: False
                    flatten: False
                    sequence_type: amino_acid
                    region_type: imgt_cdr3

            one_hot_positional:
                OneHot:
                    use_positional_info: True
                    distance_to_seq_middle: 3
                    flatten: False
                    sequence_type: nucleotide



ProtT5
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Encoder based on a pretrained protein language model by Elnaggar et al. 2021. The used transformer model is
"Rostlab/prot_t5_xl_half_uniref50-enc".

Original publication:
Elnaggar, A., Heinzinger, M., Dallago, C., Rihawi, G., Wang, Y., Jones, L., Gibbs, T., Feher, T.,
Angerer, C., Steinegger, M., Bhowmik, D., & Rost, B. (2021). ProtTrans: Towards Cracking the Language of
Life's Code Through Self-Supervised Deep Learning and High Performance Computing (No. arXiv:2007.06225).
arXiv. https://doi.org/10.48550/arXiv.2007.06225

Original GitHub repository with license information: https://github.com/agemagician/ProtTrans

**Dataset type:**

- SequenceDatasets

- ReceptorDatasets

- RepertoireDatasets

**Specification arguments:**

- region_type (RegionType): Which part of the receptor sequence to encode. Defaults to IMGT_CDR3.

- device (str): Which device to use for model inference - 'cpu', 'cuda', 'mps' - as defined by pytorch.
  Defaults to 'cpu'.

- num_processes (int): Number of processes to use for parallel processing. Defaults to 1.

- batch_size (int): The number of sequences to encode at the same time. This could have large impact on memory usage.
  If memory is an issue, try with smaller batch sizes. Defaults to 4096.

- scale_to_zero_mean (bool): Whether to scale the embeddings to zero mean. Defaults to True.

- scale_to_unit_variance (bool): Whether to scale the embeddings to unit variance. Defaults to True.

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        encodings:
            my_prot_t5_encoder:
                ProtT5::
                    region_type: IMGT_CDR3
                    device: cpu
                    num_processes: 1
                    batch_size: 4096



SequenceAbundance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


This encoder represents the repertoires as vectors where:

- the first element corresponds to the number of label-associated clonotypes
- the second element is the total number of unique clonotypes

To determine what clonotypes (with features defined by comparison_attributes) are label-associated, one-sided Fisher's exact test is used.

The encoder also writes out files containing the contingency table used for Fisher's exact test,
the resulting p-values, and the significantly abundant sequences
(use :py:obj:`~immuneML.reports.encoding_reports.RelevantSequenceExporter.RelevantSequenceExporter` to export these sequences in AIRR format).

Reference: Emerson, Ryan O. et al.
‘Immunosequencing Identifies Signatures of Cytomegalovirus Exposure History and HLA-Mediated Effects on the T Cell Repertoire’.
Nature Genetics 49, no. 5 (May 2017): 659–65. `doi.org/10.1038/ng.3822 <https://doi.org/10.1038/ng.3822>`_.

Note: to use this encoder, it is necessary to explicitly define the positive class for the label when defining the label
in the instruction. With positive class defined, it can then be determined which sequences are indicative of the positive class.
For full example of using this encoder, see :ref:`Reproduction of the CMV status predictions study`.

**Dataset type:**

- RepertoireDatasets

.. note::

    This encoder is computationally intensive and may require a large amount of memory and time to run. Use
    CompAIRRSequenceAbundance encoder instead for more efficient computation and for the same functionality.


**Specification arguments:**

- comparison_attributes (list): The attributes to be considered to group receptors into clonotypes. Only the fields specified in
  comparison_attributes will be considered, all other fields are ignored. Valid comparison value can be any
  repertoire field name (e.g., as specified in the AIRR rearrangement schema).

- p_value_threshold (float): The p value threshold to be used by the statistical test.

- sequence_batch_size (int): The number of sequences in a batch when comparing sequences across repertoires, typically 100s of thousands.
  This does not affect the results of the encoding, only the speed. The default value is 1.000.000

- repertoire_batch_size (int): How many repertoires will be loaded at once. This does not affect the result of the encoding, only the speed.
  This value is a trade-off between the number of repertoires that can fit the RAM at the time and loading time from disk.


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        encodings:
            my_sa_encoding:
                SequenceAbundance:
                    comparison_attributes:
                        - cdr3_aa
                        - v_call
                        - j_call
                    p_value_threshold: 0.05
                    sequence_batch_size: 100000
                    repertoire_batch_size: 32



ShannonDiversity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


ShannonDiversity encoder calculates the Shannon diversity index for each repertoire in a dataset. The diversity is
computed as:

.. math::

    diversity = - \sum_{i=1}^{n} p_i \log(p_i)

where :math:`p_i` is the clonal count for each unique sequence in the repertoire (from duplicate_count field)
divided by the total clonal counts, and :math:`n` is the total number of clonotypes (sequences) in the repertoire.


**Dataset type:**

- RepertoireDataset

**Specification arguments:**

No arguments are needed for this encoder.

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        encodings:
            shannon_div_enc: ShannonDiversity



SimilarToPositiveSequence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


A simple baseline encoding, to be used in combination with :py:obj:`~immuneML.ml_methods.BinaryFeatureClassifier.BinaryFeatureClassifier` using keep_all = True.
This encoder keeps track of all positive sequences in the training set, and ignores the negative sequences.
Any sequence within a given hamming distance from a positive training sequence will be classified positive,
all other sequences will be classified negative.

**Dataset type:**

- SequenceDatasets


**Specification arguments:**

- hamming_distance (int): Maximum number of differences allowed between any positive sequence of the training set and a
  new observed sequence in order for the observed sequence to be classified as 'positive'.

- compairr_path (Path): optional path to the CompAIRR executable. If not given, it is assumed that CompAIRR
  has been installed such that it can be called directly on the command line with the command 'compairr',
  or that it is located at /usr/local/bin/compairr.

- ignore_genes (bool): Only used when compairr is used. Whether to ignore V and J gene information. If False, the V and J genes between two sequences
  have to match for the sequence to be considered 'similar'. If True, gene information is ignored. By default, ignore_genes is False.

- threads (int): The number of threads to use for parallelization. This does not affect the results of the encoding, only the speed.
  The default number of threads is 8.

- keep_temporary_files (bool): whether to keep temporary files, including CompAIRR input, output and log files, and the sequence
  presence matrix. This may take a lot of storage space if the input dataset is large. By default temporary files are not kept.


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        encodings:
            my_sequence_encoder:
                SimilarToPositiveSequenceEncoder:
                    hamming_distance: 2


TCRBert
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


TCRBertEncoder is based on `TCR-BERT <https://github.com/wukevin/tcr-bert/tree/main>`_, a large language model
trained on TCR sequences. TCRBertEncoder embeds TCR sequences using either of the pre-trained models provided on
HuggingFace repository.

Original publication:
Wu, K. E., Yost, K., Daniel, B., Belk, J., Xia, Y., Egawa, T., Satpathy, A., Chang, H., & Zou, J. (2024).
TCR-BERT: Learning the grammar of T-cell receptors for flexible antigen-binding analyses. Proceedings of the
18th Machine Learning in Computational Biology Meeting, 194–229. https://proceedings.mlr.press/v240/wu24b.html

**Dataset type:**

- SequenceDataset

- ReceptorDataset

- RepertoireDataset

**Specification arguments:**

- model (str): The pre-trained model to use (huggingface model hub identifier). Available options are 'tcr-bert'
  and 'tcr-bert-mlm-only'.

- layers (list): The hidden layers to use for encoding. Layers should be given as negative integers, where -1
  indicates the last representation, -2 second to last, etc. Default is [-1].

- method (str): The method to use for pooling the hidden states. Available options are 'mean', 'max'',
  'cls', and 'pool'. Default is 'mean'. For explanation of the methods, see GitHub repository of TCR-BERT.

- batch_size (int): The number of sequences to encode at the same time. This could have large impact on memory usage.
  If memory is an issue, try with smaller batch sizes. Defaults to 4096.

- scale_to_zero_mean (bool): Whether to scale the embeddings to zero mean. Defaults to True.

- scale_to_unit_variance (bool): Whether to scale the embeddings to unit variance. Defaults to True.

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        encodings:
            my_tcr_bert_encoder: TCRBert



TCRdist
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Encodes the given ReceptorDataset as a distance matrix between all receptors, where the distance is computed using TCRdist from the paper:
Dash P, Fiore-Gartland AJ, Hertz T, et al. Quantifiable predictive features define epitope-specific T cell receptor repertoires.
Nature. 2017; 547(7661):89-93. `doi:10.1038/nature22383 <https://www.nature.com/articles/nature22383>`_.

For the implementation, `TCRdist3 <https://tcrdist3.readthedocs.io/en/latest/>`_ library was used (source code available
`here <https://github.com/kmayerb/tcrdist3>`_).

**Dataset type:**

- ReceptorDataset

- SequenceDataset


**Specification arguments:**

- cores (int): number of processes to use for the computation

- cdr3_only (bool): whether to use only cdr3 or also v gene; if set to false, encoding will only compute the distances
  between the CDR3 regions of the receptors


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        encodings:
            my_tcr_dist_enc:
                TCRdist:
                    cores: 4
                    cdr3_only: false # default tcrdist behavior



Word2Vec
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



Word2VecEncoder learns the vector representations of k-mers based on the context (receptor sequence).
Similar idea was discussed in: Ostrovsky-Berman, M., Frankel, B., Polak, P. & Yaari, G.
Immune2vec: Embedding B/T Cell Receptor Sequences in ℝN Using Natural Language Processing. Frontiers in Immunology 12, (2021).

This encoder relies on gensim's implementation of Word2Vec and KmerHelper for k-mer extraction. Currently it works on amino acid level.

**Dataset type:**

- SequenceDatasets

- RepertoireDatasets


**Specification arguments:**

- vector_size (int): The size of the vector to be learnt.

- model_type (:py:obj:`~immuneML.encodings.word2vec.model_creator.ModelType.ModelType`):  The context which will be
  used to infer the representation of the sequence.
  If :py:obj:`~immuneML.encodings.word2vec.model_creator.ModelType.ModelType.SEQUENCE` is used, the context of
  a k-mer is defined by the sequence it occurs in (e.g. if the sequence is CASTTY and k-mer is AST,
  then its context consists of k-mers CAS, STT, TTY)
  If :py:obj:`~immuneML.encodings.word2vec.model_creator.ModelType.ModelType.KMER_PAIR` is used, the context for
  the k-mer is defined as all the k-mers that within one edit distance (e.g. for k-mer CAS, the context
  includes CAA, CAC, CAD etc.).
  Valid values are `SEQUENCE`, `KMER_PAIR`.

- k (int): The length of the k-mers used for the encoding.

- epochs (int): for how many epochs to train the word2vec model for a given set of sentences (corresponding to epochs parameter in gensim package)

- window (int): max distance between two k-mers in a sequence (same as window parameter in gensim's word2vec)


**YAML pecification:**

.. highlight:: yaml
.. code-block:: yaml

    definitions:
        encodings:
            encodings:
                my_w2v:
                    Word2Vec:
                        vector_size: 16
                        k: 3
                        model_type: SEQUENCE
                        epochs: 100
                        window: 8


