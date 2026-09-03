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
An introduction to the YAML specification can be found here:

.. toctree::
  :caption: General introduction
  :maxdepth: 1

  yaml_specs/how_to_specify_an_analysis_with_yaml


.. toctree::
  :caption: Detailed parameter descriptions
  :maxdepth: 1

  yaml_specs/datasets
  yaml_specs/encodings
  yaml_specs/ml_methods
  yaml_specs/reports
  yaml_specs/preprocessings
  yaml_specs/simulation
  yaml_specs/instructions


