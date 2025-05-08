immuneML data model
=====================


.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML dev docs: data model
   :twitter:image: https://docs.immuneml.uio.no/_images/data_model_architecture.png


immuneML works with adaptive immune receptor sequencing data.
Internally, the classes and data structures used to represent this data adheres to the `AIRR Rearrangement Schema <https://docs.airr-community.org/en/stable/datarep/rearrangements.html>`_,
although it is possible to import data from a wider variety of common formats.

Most immuneML analyses are based on the amino acid CDR3 junction.
Some analyses also use the V and J gene name ('call') information.
While importing of full-length (V + CDR3 + J) sequences is supported,
there are no functionalities in immuneML designed for analysing sequences
at that level.

immuneML data model supports three types of datasets that can be used for analyses:

#. Repertoire dataset (:py:obj:`~immuneML.data_model.dataset.RepertoireDataset.RepertoireDataset`) - each example in the dataset is a large set of AIR sequences which are typically derived from one subject (individual).
#. Receptor dataset (:py:obj:`~immuneML.data_model.dataset.ReceptorDataset.ReceptorDataset`) - each example is one paired-chain receptor consisting of two AIR sequences (e.g., TCR alpha-beta, or IGH heavy-light).
#. Sequence dataset (:py:obj:`~immuneML.data_model.dataset.SequenceDataset.SequenceDataset`) - each example is one single AIR sequence chain.


A single AIR rearrangement is represented by a :py:obj:`~immuneML.data_model.receptor.receptor_sequence.ReceptorSequence.ReceptorSequence` class.
A sequence dataset contains a set of such ReceptorSequence objects. A receptor dataset contains a set of
:py:obj:`~immuneML.data_model.receptor.receptor.Receptor.Receptor` objects, which contain two ReceptorSequences each.
Relevant shared code for Sequence- and ReceptorDatasets can be found in the :py:obj:`~immuneML.data_model.dataset.ElementDataset.ElementDataset` class.
A Repertoire dataset contains a set of :py:obj:`~immuneML.data_model.repertoire.receptor.Repertoire.Repertoire` objects, which
each contain a set of ReceptorSequence objects.
