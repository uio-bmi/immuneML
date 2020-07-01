.. ImmuneML documentation master file, created by
   sphinx-quickstart on Mon Jul 29 19:02:08 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ImmuneML's documentation!
====================================

Welcome to immuneML. To get started with using ImmuneML, please consult our documentation: :ref:`Installing immuneML`.

ImmuneML is a platform for machine learning-based immune repertoire analysis and classification. It is available as `open source code at github <https://github.com/uio-bmi/ImmuneML>`_,
as a pip package, as a Docker image, as an Amazon cloud virtual machine image, through a REST API and through a `Galaxy Portal <http://129.240.189.178:8080/>`_.

In short, ImmuneML is typically run by specifying:

- A set of repertoire files to be analyzed,

- A tabular file describing metadata like disease classes for the repertoires,

- A specification of the analysis to be performed as a YAML file.

The most central aspect is the YAML-based specification that allows a very high flexibility in defining analyses.

To get a feel for this, you could either consult our :ref:`Quickstart` guide or get a grasp of the overall structure and possibilities of the analysis
specification with :ref:`How to specify an analysis with YAML`. After this, you can consult more detailed :ref:`Tutorials` for specific use cases like
generation of random repertoire data, simulating disease-associated signals, performing descriptive analyses, and not the least for training and
assessing classifiers.

To ensure that your file is valid YAML, you can write/validate your specification in a local or `online <https://jsonformatter.org/yaml-validator>`_ editor with YAML support.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   quickstart
   installation
   tutorials
   galaxy
   specification
   FAQ
   developer_documentation


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
