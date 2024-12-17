ML basics: Training classifiers with the simplified Galaxy interface
==================================================================================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML & Galaxy: train immune repertoire classifier with simplified interface
   :twitter:description: See tutorials on how to train immune repertoire classifiers using the simplified Galaxy interface.
   :twitter:image: https://docs.immuneml.uio.no/_images/repertoire_classification_overview.png





This page provides the documentation for the Galaxy tools `Train Receptor Classifier (Simplified Interface) <https://galaxy.immuneml.uiocloud.no/root?tool_id=immuneml_train_receptor_classifier>`_
and `Train Repertoire Classifier (Simplified Interface)  <https://galaxy.immuneml.uiocloud.no/root?tool_id=immuneml_train_repertoire_classifier>`_,
and serves as an introduction to immuneML users without extensive machine learning (ML) knowledge.


The purpose of this tool is to train ML models to predict a characteristic per immune repertoire, such as
a disease status; or an immune receptor, such as antigen binding.
One or more ML models are trained to classify repertoires based on the information within the sets of CDR3 sequences. Finally, the performance
of the different methods is compared.


Basic terminology
-----------------

In the context of ML, the characteristics to predict per repertoire are called **labels** and the values that these labels can take on are **classes**.
One could thus have a label named ‘CMV_status’ with possible classes ‘positive’ and ‘negative’ for repertoires, or
‘epitope’ with possible classes ‘binding_CMV’ and ‘not_binding_CMV’ for receptors.

For repertoire datasets, the labels and classes must be present in the metadata file, in columns where the header and
values correspond to the label and classes respectively. For receptor datasets, the labels are present as additional columns in the input file.

.. figure:: ../_static/images/metadata_repertoire_classification.png
  :width: 70%

  Metadata for repertoire classification

When training an ML model, the goal is for the model to learn **signals** within the data which discriminate between the different classes. An ML model
that predicts classes is also referred to as a **classifier**. A signal can have a variety of definitions, including the presence of specific receptors,
groups of similar receptors or short CDR3 subsequences in an immune repertoire. Our assumptions about what makes up a ‘signal’ determines how we
should represent our data to the ML model. This representation is called **encoding**. In this tool, the encoding is automatically chosen based on
the user's assumptions about the dataset.

Receptor classification overview
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../_static/images/receptor_classification_overview.png
  :width: 70%

  An overview of the components of the immuneML receptor classification tool.

ImmuneML reads in receptor data with labels (+ and -), encodes the data, trains user-specified ML models and summarizes
the performance statistics per ML method.
**Encoding:** position dependent and invariant encoding are shown. The specificity-associated subsequences are highlighted
with color. The different colors represent independent elements of the antigen specificity signal. Each color represents
one subsequence, and position dependent subsequences can only have the same color when they occur in the same position,
although different colors (i.e., nucleotide or amino acid sequences) may occur in the same position.
**Training:** the training and validation data is used to train ML models and find the optimal hyperparameters through
5-fold cross-validation. The test set is left out and is used to obtain a fair estimate of the model performance.


Repertoire classification overview
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../_static/images/repertoire_classification_overview.png
  :width: 70%

  An overview of the components of the immuneML repertoire classification tool.

immuneML reads in repertoire data with labels (+ and -), encodes the
data, trains user-specified ML models and summarizes the performance statistics per ML method.
**Encoding:** different forms of encoding are shown; full sequence encoding and position dependent and invariant subsequence encoding.
The disease-associated sequences or sub-sequences are highlighted with color. The different colors represent independent elements of the disease signal.
Each color represents one (sub)sequence, and position dependent subsequences can only have the same color when they occur in the same position,
although different colors (i.e., nucleotide or amino acid sequences) may occur in the same position.
**Training:** the training and validation data is used to train ML models and find the optimal hyperparameters through 5-fold cross-validation.
The test set is left out and is used to obtain a fair estimate of the model performance.


Encoding
---------

The simplest encoding represents an immune repertoire based on the full CDR3 sequences that it contains. This means the ML models will learn to look
at which CDR3 sequences are more often present in the ‘positive’ or ‘negative’ classes. It also means that two similar (non-identical) CDR3 sequences
are treated as independent pieces of information; if a particular sequence often occurs in diseased repertoires, then finding a similar sequence in a
new repertoire is no evidence for this repertoire also being diseased. Note that this type of encoding does not make sense for immune
receptor datasets, as it would mean the ML model can only learn that receptors are binders or non-binders if these exact same receptors were observed in the
training dataset.

Other encoding variants are based on shorter subsequences (e.g., 3 – 5 amino acids long, also referred to as k-mers) in the CDR3 regions of an immune repertoire. With this
encoding, the CDR3 regions are divided into overlapping subsequences and the (disease) signal may be characterized by the presence or absence of
certain sequence motifs in the CDR3 regions. Here, two similar CDR3 sequences are no longer independent, because they contain many identical subsequences.
This type of encoding can be used both for repertoires and receptors.
A graphical representation of how a CDR3 sequence can be divided into k-mers, and how these k-mers can relate to specific positions in a 3D immune receptor
(here: antibody) is shown in this figure:

.. image:: ../_static/images/3mer_to_3d.png
  :width: 60%


The subsequences may be position-dependent or invariant. Position invariant means that if a subsequence, e.g., ‘EDNA’ occurs in different positions
in the CDR3 it will still be considered the same signal. This is not the case for position dependent subsequences, if ‘EDNA’ often occurs in the
beginning of the CDR3 in diseased repertoires, then finding ‘EDNA’ in the end of a CDR3 in a new repertoire will be considered unrelated. Positions
are determined based on the IMGT numbering scheme.

Finally, it is possible to introduce gaps in the encoding of subsequences (not shown in the Figure). In this case, a motif is defined by two
subsequences separated by a region of varying nucleotide or amino acid length. Thus, the subsequences ‘EDNA’, ‘EDGNA’ and ‘EDGAGAGNA’ may all be
considered to be part of the same motif: ‘ED’ followed by ‘NA’ with a gap of 0 – 5 amino acids in between.

Note that in any case, the (sub)sequences that are associated with the ‘positive’ class may still be present in the ‘negative’ class, albeit at a lower rate.

Training a machine learning model
----------------------------------

Training an ML model means optimizing the **parameters** for the model with the goal of predicting the correct class of an (unseen) immune repertoire.
Different ML methods require different procedures for training. In addition to the model parameters there are the **hyperparameters**, which
do not directly change the predictions of a model, but they control the learning process (for example: the learning speed).

The immune repertoires are divided into sets with different purposes: the training and validation sets are used for finding the optimal parameters
and hyperparameters respectively. The test set is held out, and is only used to estimate the performance of a trained model.

In this tool, a range of plausible hyperparameters have been predefined for each ML method. The optimal hyperparameters are found by splitting the
training/validation data into 5 equal portions, where 4 portions are used to train the ML model (with different hyperparameters) and the remaining
portion is used to validate the performance of these hyperparameter settings. This is repeated 5 times such that each portion has been used for
validation once. With the best hyperparameters found in the 5 repetitions, a final model is trained using all 5 portions of the data. This procedure
is also referred to as 5-fold cross-validation. Note that this 5-fold cross-validation is separate from the number of times the splitting into
training + validation and testing sets is done (see the overview figure).

Finally, the whole process is repeated one or more times with different randomly selected repertoires in the test set, to see how robust the performance
of the ML methods is. The number of times to repeat this splitting into training + validation and test sets is determined in the last question.





