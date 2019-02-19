############
Data model
############

.. toctree::
   :maxdepth: 2


The ImmuneML's data model consists of:

*   Sequence class
*   Repertoire class
*   Dataset class

Sequence
========

The Sequence class contains all information about a single immune receptor sequence. For example, a sequence can refer to
a CDR3 sequence. The information stored about a sequence are:

*   amino acid sequence,
*   nucleotide sequence,
*   id,
*   annotation,
*   metadata,

**Id** is any unique string that will unambiguously identify the sequence.

**Sequence metadata** contains information about V gene, J gene, chain, sequence count, region and frame type. They can be used for
further analysis. For instance, when analyzing sequences, those that have "Out" or "Stop" frame types would be discarded.

**Sequence annotation** is used for simulation purposes. In cases when sequences are modified to include an artificial
disease signal, this object will store the information on how exactly the sequence was modified.

Repertoire
==========

The Repertoire class contains information about a single immune repertoire. A repertoire object consists of:

*   a list of sequences and
*   a metadata object.

A list of **sequences** includes all sequences coming from the same person. Each sequence is an instance of a :ref:`Sequence` object.

**Repertoire metadata** contains information about a repertoire. That information is modeled by information about a sample
and, in case of simulation, list of modifications to sequences in the repertoire.

**Sample** is defined by a unique identifier, an optional name (in case the unique identifier is not descriptive enough)
and optional other parameters. Examples of such parameters could be:

*   date of the experiment,
*   age of the patient,
*   known diseases of the patient etc.

In case only sequences should be analyzed, regardless of the repertoires, the *Sequence*, *Repertoire* and *Dataset* classess
should still be used, but it is necessary then to make each repertoire to consist of only one sequence. Everything else in the
analysis, except where noted, can be used in the same manner as when the analysis has repertoires consisting of a bulk of
sequences.

Dataset
=======

Dataset class models a list of repertoires. It contains the following information:

*   a unique identifier,
*   a list of repertoires,
*   a list of filenames,
*   encoded repertoires and
*   dataset parameters.

If not set manually by the user, the unique identifier is automatically generated.

A list of repertoires is a list of objects of :ref:`Repertoire` class. In case the repertoires occupy too much memory,
and cannot be loaded all at once, the dataset contains a list of paths to each repertoire file for the dataset. Each
repertoire then is loaded from the file as needed, thus avoiding memory issues.

Encoded repertoires are used for machine learning setting. Since machine learning algorithms cannot work with the data
in their original format, they are encoded so that they can be further analyzed. Examples of the encoding include k-mer
decomposition and encoded a repertoire by k-mer frequencies, vector embeddings on repertoire level and others.

Dataset parameters are an instance of **DatasetParams** class and include the following:

*   number of repertoires in the dataset,
*   path to dataset file,
*   name of the encoding type,
*   a list of parameters available in samples for each repertoire (e.g. date, age, disease).



