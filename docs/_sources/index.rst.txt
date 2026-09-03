.. immuneML documentation master file, created by
   sphinx-quickstart on Mon Jul 29 19:02:08 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the immuneML documentation!
======================================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML documentation and tutorials
   :twitter:description: immuneML is an open-source software platform for machine learning analysis of adaptive immune receptor repertoires, available as a Python library, through Galaxy and as a Docker image. On this website, you can browse the platform's documentation and tutorials.
   :twitter:image: https://docs.immuneml.uio.no/_images/receptor_classification_overview.png

immuneML is a platform for machine learning-based analysis and classification of adaptive immune receptors and
repertoires (AIRR). To **get started using immuneML right away**, check out our :ref:`Quickstart` tutorial.

immuneML can be used for:

- **Exploratory analysis of datasets** such as dataset overview, statistical analyses, visualizations to get the
  overview of the data;

- **Clustering analysis of datasets** to examine whether the examples form any clusters, how stable the clusters are,
  how much the clusters correspond to any of the external labels available;

- **Training classification ML models** for repertoire classification (e.g., disease prediction) or receptor sequence
  classification (e.g., antigen binding prediction), and applying them to new datasets with unknown class labels;

- **Simulating datasets** for ML model benchmarking with known ground truth immune signals using LIgO;

- **Training generative ML models** of receptor sequences and evaluating synthetic sequences across a range of
  characteristics.

The starting point for any immuneML analysis is the YAML specification file. In this file, the settings of the analysis
components are defined (also known as :code:`definitions`), which are shown in six different colors in the figure below. Additionally, the YAML file
describes one or more :code:`instructions`, which corresponds to one of the applications listed above (and some additional instructions).


.. figure:: _static/images/definitions_instructions_overview.png
   :alt: immuneML usage overview

   *An overview of immuneML usage: analysis components and instructions are specified in a YAML file. Each use case corresponds to a different instruction. The results of the instructions are summarized and presented in an HTML file.*


Getting started
-------------------

If you want to **use immuneML locally**, see :ref:`Installing immuneML`.

To become familiar with the **YAML specification**, you can find a concrete example in our :ref:`Quickstart` guide, or read about the overall YAML structure and options in :ref:`How to specify an analysis with YAML`.

Alternatively, to **run immuneML in a web browser**, go to our `Galaxy Portal <https://avant.immuneml.uiocloud.no/>`_.
Here, we offer the same functionalities as in the command-line interface (using YAML specifications), and in addition simplified button-based interfaces for training classifiers.
See the :ref:`immuneML & Galaxy` tutorials for more information.

immuneML can be applied to a wide variety of **use cases**. To help you get started, we offer :ref:`Tutorials` for some common applications (e.g., how to train models, or how to simulate synthetic data for benchmarking).
For more experienced users who want to customize their analysis and are wondering about all the possible **analysis components** and their options, you can find the complete list and
documentation under :ref:`YAML specification`.

Our open-source code can be found on `GitHub <https://github.com/uio-bmi/ImmuneML>`_ :)

Previous versions
-------------------

Documentation for previous immuneML versions can be found here:

- `v2.2.6 <https://docs.immuneml.uio.no/v2.2.6/>`_
- `v2.1.2 <https://docs.immuneml.uio.no/v2.1.2/>`_
- `v2.1.0 <https://docs.immuneml.uio.no/v2.1.0/>`_
- `v2.0.4 <https://docs.immuneml.uio.no/v2.0.4/>`_
- `v1.2.5 <https://docs.immuneml.uio.no/v1.2.5/>`_


.. toctree::
   :maxdepth: 1
   :hidden:

   quickstart
   installation
   specification
   tutorials
   galaxy
   usecases
   troubleshooting
   developer_docs

