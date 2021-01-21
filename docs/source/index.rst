.. immuneML documentation master file, created by
   sphinx-quickstart on Mon Jul 29 19:02:08 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the immuneML documentation!
======================================

immuneML is a platform for machine learning-based analysis and classification of adaptive immune receptors and
repertoires. immuneML can be used for:

- **Training ML models** for repertoire classification (e.g., disease prediction) or receptor sequence
  classification (e.g., antigen binding prediction). In immuneML, the performance of different machine learning (ML)
  settings can be compared by nested cross-validation. These ML settings consist of data preprocessing steps, encodings
  and ML models and their hyperparameters.

- **Exploratory analysis of datasets** by applying preprocessing and encoding, and plotting descriptive statistics without training ML models.

- **Simulating** immune events, such as disease states, into experimental or synthetic repertoire datasets.
  By implanting known immune signals into a given dataset, a ground truth benchmarking dataset is created. Such a dataset
  can be used to test the performance of ML settings under known conditions.

- **Applying trained ML models** to new datasets with unknown class labels.

The starting point for any immuneML analysis is the YAML specification file. In this file, the settings of the analysis
components are defined, which are shown in six different colors in the figure below. Additionally, the YAML file
describes an *instruction*, which corresponds to one of the applications listed above (and some additional instructions).


.. figure:: _static/images/definitions_instructions_overview.png
   :alt: immuneML usage overview

   An overview of immuneML usage: analysis components and instructions are specified in a YAML file. Each use case corresponds to a different instruction. The results of the instructions are summarized and presented in an HTML file.


To **get started using immuneML right away**, you can check out our `Galaxy Portal <http://immunohub01.hpc.uio.no:8080/>`_.
Here, we offer the same functionalities as in the command-line interface, and in addition simplified button-based
interfaces for training classifiers.

If you want to **use immuneML locally**, see :ref:`Installing immuneML`.

To become familiar with the **YAML-based specification**, you can find a concrete example in our :ref:`Quickstart` guide,
or read about the overall YAML structure and options in :ref:`How to specify an analysis with YAML`.

- Note that the components of the YAML specification are the same in the Galaxy portal and in the command-line interface,
  but in Galaxy, datasets must first be converted to immuneML format (see :ref:`How to make an immuneML dataset in Galaxy`).

Once you have a general understanding of the YAML specification, you can take a look at more detailed :ref:`Tutorials` for
specific use cases (e.g., how to train and assess classifiers, or how to generate synthetic immune repertoire data for benchmarking purposes).

If you are wondering about all the possible **analysis components** and their settings, you can find the complete list and
documentation under :ref:`YAML specification`.

Our open-source code can be found on `GitHub <https://github.com/uio-bmi/ImmuneML>`_ :)


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   quickstart
   installation
   tutorials
   galaxy
   specification
   usecases
   FAQ
   developer_docs


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
