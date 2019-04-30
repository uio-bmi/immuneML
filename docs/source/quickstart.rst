#################
Quickstart guide
#################

This guide will show how to use the ImmuneML and run a simple analysis on simulated data.

Analysis example
=================

The goal of the analysis is to predict whether an individual has a disease or not (i.e. to predict the immune state
of an individual). This prediction will be made based on the machine learning analysis of the immune repertoire of a
person. The person's immune repertoire consists of all immune receptor sequences in the individual, some of which are
specific for the disease in question.

In order to be able to predict if an individual has a disease with a machine learning algorithm, it is necessary to
train the algorithm with known examples. The known examples are the repertoires of people who have the disease and the
repertoires of the people who do not have the disease. With these examples at hand, the algorithm can learn to
distinguish between repertoires with the disease and those without, thus yielding a useful approach to prediction.

To be able to learn something from the immune repertoires, machine learning algorithms require the repertoires to have a
suitable representation. In this analysis, a suitable representation will be made by calculating k-mer frequencies in a
repertoire. For instance, given a sequence ``CASSRTY``, then the resulting 3-mers are ``CAS``, ``ASS``, ``SSR``,
``SRT``, ``RTY``. To obtain this representation, each sequence in a repertoire is split into overlapping 3-mers and the
frequency of each possible 3-mer is calculated for the repertoire.

The workflow of the quickstart analysis
=======================================

The analysis will consist of the following steps:

1.  Simulation of the repertoires
2.  Signal implantation in some of the repertoires
3.  Splitting the repertoires to train and test set
4.  Representing repertoires by their k-mer frequencies
5.  Training a machine learning algorithm
6.  Testing the algorithm

The repertoires are simulated by *RandomDatasetGenerator*. All repertoires will consist of a predefined number of immune
receptor sequences which will be randomly generated from the list of available amino acids. Defined in this manner, all
repertoires are generated in the same way and none are disease-specific.

In order to simulate the fact that some of the repertoires (and corresponding individuals) have encountered a certain
disease, a specific signal will be implanted into some of the repertoires. The disease-specific signal, as defined in the
:ref:`Simulation model` will consist of one motif which will be instantiated using the *IdentityMotifInstantiation*.
This means that when for a disease (i.e. for the signal) a motif with seed ``CAS`` is defined, ``CAS`` is what will always
be implanted into the repertoires.

The repertoires defined in this manner will be split to train and test sets which are necessary to evaluate the performance
of machine learning algorithms and provide the best possible estimate of the expected prediction error when the algorithm
is used on new examples.

The split datasets will then be transformed to suitable representation based on k-mer frequencies.

Three different machine learning algorithms will be trained to perform the prediction if the patient has a disease or not.
Those algorithms are support vector machine, logistic regression and random forest.

Each of these algorithms will then be tested and their prediction accuracy measured on the test dataset. The reported
accuracy is the expected prediction accuracy of classification of new repertoires.

Performing the analysis
========================

Run specification
-----------------

In the quickstart.py, a Quickstart class is defined that will perform the analysis. The configuration for the analysis
is given as a Python dictionary in the following manner:

.. code-block:: python

    Quickstart.perform_analysis({
        "repertoire_count": 400, # number of repertoires to create
        "sequence_count": 500, # number of sequences in each repertoire
        "receptor_type": "TCR",
        "result_path": "../../../simulation_results/", # store results in the root project folder under simulation_results
        "ml_methods": ["LogisticRegression", "SVM", "RandomForest"],
        "training_percentage": 0.7,
        "cv": 10, # choose hyperparameters for ML algorithms by 10-fold cross-validation
        "encoder": "KmerFrequencyEncoder",
        "encoder_params": {
            "sequence_encoding": "continuous_kmer",
            "k": 3,
            "reads": "unique",
            "normalization_type": "relative_frequency"
        },
        "simulation": {
            "motifs": [
                {
                    "id": "motif1",
                    "seed": "CAS",
                    "instantiation": "identity"
                }
            ],
            "signals": [
                {
                    "id": "signal1",
                    "motifs": ["motif1"],
                    "implanting": "healthy_sequences"
                }
            ],
            "implanting": [{
                "signals": ["signal1"],
                "repertoires": 0.5, # the percentage of repertoires with implanted signal
                "sequences": 0.2 # the percentage of sequences with the given signal in the repertoire chosen for implanting
            }]
        }
    })


Prerequisites
-------------

To be able to run the code, first install the Python packages listed below. Installation instructions are given using pip.
More details and installation instructions can be found on the corresponding package's websites.

1.  Numpy_

.. code-block:: RST

    $ pip install numpy

2.  Sklearn_

.. code-block:: RST

    $ pip install sklearn

3.  Gensim_

.. code-block:: RST

    $ pip install gensim

Run the analysis
----------------

To perform the analysis described in the previous sections, do the following:

1.  Clone the GitHub repository
2.  Navigate to the ImmuneML folder from the cloned repository
3.  Execute the following line in the terminal:

.. code-block:: RST

    $ python3 source/workflows/processes/quickstart.py

.. _Numpy: http://www.numpy.org/
.. _Sklearn: https://scikit-learn.org/stable/index.html
.. _Gensim: https://radimrehurek.com/gensim/

