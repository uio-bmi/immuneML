#################
Simulation model
#################

.. toctree::
   :maxdepth: 2


To enable a simulation in which the prediction of the immune status can be performed for synthetic data, ImmuneML
introduces the simulation model.

Synthetic data can be defined in different ways:

*   generating synthetic sequences so that some of them are specific to a user-defined disease,
*   generating synthetic sequences so that some of them are specific to a user-defined disease and
    grouping them in repertoires and
*   extracting publicly available sequences and changing them so that they are specific to a user-defined disease,
    grouping them in repertoires.

Changing the receptor sequences so that they are disease-specific is performed in the following way: a disease
is represented by a *signal* and that signal is implanted in a sequence. A signal, in order to accommodate variety of
paratopes and reflect biological complexity, can consist of multiple motifs. A motif is a probabilistically defined
sequence of amino acids or nucleotides. According to a user-determined strategy, a concrete instance of a motif
(an exact sequence, such as ``TYQRTRALV`` for influenza [1]_) is created and implanted into the immune receptor sequence.

Signal
======

The **Signal** class corresponds to a disease. An object of the Signal class consists of:

*   a unique identifier,
*   a list of motifs,
*   a strategy for implanting

Motif
=====

The **Motif** class includes a seed and defines a way of creating a concrete sequence (a motif instance) from the seed.
It consists of:

*   a unique identifier,
*   seed,
*   position weights,
*   alphabet weights,
*   a strategy for creating instances of the motif.

*Seed* is a starting point for creating a sequence which will be implanted into an immune receptor sequence. It is a
sequence of amino acid or nucleotide one letter codes, such as ``CAS`` or ``ACT``, respectively. Since the epitopes can
have a gap within the sequence, the position of the gap can also be marked in the seed with ``/`` character. For seed
``C/AS``, the gap is located after ``C`` and before ``AS``.

Position weights are probabilities that a position in the seed will be changed when creating a specific instance of the
motif. If a seed is ``CAS``, then position weight 1 for the first position means that when creating a motif instance, the
first letter (``C``) will be changed if changes from the seed are allowed.

Alphabet weights are probabilities that a given letter (either an amino acid or a nucleotide) will be chosen to replace
the letter in the seed sequence. For amino acid sequences, alphabet consists of 20 letters corresponding to each amino acid
one letter code. If weight for ``A`` is specified to be 1, then when choosing a letter to change one of the letters from
the seed, letter ``A`` will always be chosen. If alphabet weights are not specified, then all letters have equal probability
of being chosen for replacement.

Motif instance
==============

The **MotifInstance** class encapsulates a single instance of a motif and ensures it has an appropriate structure. It
consists of:

*   an instance of the motif and
*   a gap.

For example, if ``C/AS`` is the motif seed where ``/`` denotes the gap location, an instance of the motif could be ``C/AS``
and the gap size is given by the gap parameter (e.g. 2). In this case, the motif instance when implanted into sequence
``QQRTVFA`` would be ``QCRTASA``.


.. [1] Allele-specific motifs revealed by sequencing of self-peptides eluted from MHC molecules, Falk et al. 1991, Nature, https://doi.org/10.1038/351290a0