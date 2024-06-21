YAML specification
###################

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML YAML specification
   :twitter:description: See all valid options for immuneML YAML specification: for importing and simulating AIRR datasets, training ML models, choosing different encodings and reports, performing exploratory analysis.
   :twitter:image: https://docs.immuneml.uio.no/_images/receptor_classification_overview.png


All immuneML analyses are specified in a YAML-formatted file.
This file describes the different analysis components (such as which machine learning models to use),
the workflows to execute with these components, and all associated parameters.

The following pages give an introductory overview of the YAML specification file, as
well as detailed documentation of each possible parameter.


.. toctree::
  :maxdepth: 1

  yaml_specs/how_to_specify_an_analysis_with_yaml

.. toctree::
  :maxdepth: 2

  yaml_specs/yaml_parameter_details


Valid combinations of analysis components (definitions) for each dataset type can be viewed below.
These components are described by :code:`definitions`, and can be used in various different :code:`instructions` (primarily TrainMLModel and ExploratoryAnalysis).


.. image:: _static/images/analysis_paths_repertoires.png
    :alt: Analysis paths repertoires


.. image:: _static/images/analysis_paths_receptors.png
    :alt: Analysis paths receptors


.. image:: _static/images/analysis_paths_sequences.png
    :alt: Analysis paths sequences





