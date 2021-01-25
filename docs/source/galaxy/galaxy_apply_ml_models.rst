How to apply previously trained ML models to a new AIRR dataset in Galaxy
=========================================================================

After having trained ML models to a given dataset, these models can be applied to a new dataset using the Galaxy tool `Apply machine learning models to new data <https://galaxy.immuneml.uio.no/root?tool_id=immuneml_apply_ml_model>`_.
If you instead want to train new ML models, see the tutorials for training ML models for
:ref:`receptor <How to train immune receptor classifiers using the easy Galaxy interface>` and :ref:`repertoire <How to train immune repertoire classifiers using the easy Galaxy interface>`
classification using the easy Galaxy interfaces, or the more versatile :ref:`YAML-based tool for training ML models <How to train ML models in Galaxy>`.

Creating the YAML specification
---------------------------------------------
This Galaxy tool takes as input an immuneML dataset from the Galaxy history, a model training output .zip, and a YAML specification file.

The YAML specification should use the :ref:`MLApplication` instruction. The .zip file contains all information immuneML needs to
apply the same preprocessing and encoding as to the original dataset, and to make predictions using the same ML model.
More details are explained in the tutorial :ref:`How to apply previously trained ML models to a new dataset`.

When writing an analysis specification for Galaxy, it can be assumed that all selected files are present in the current working directory. A path
to an additional file thus consists only of the filename.

A complete YAML specification for applying ML models to a new dataset is shown here:


.. highlight:: yaml
.. code-block:: yaml

    definitions:
      datasets:
        dataset: # user-defined dataset name
          format: Pickle # the default format used by the 'Create dataset' galaxy tool is Pickle
          params:
            path: dataset.iml_dataset # specify the dataset name, the default name used by
                                      # the 'Create dataset' galaxy tool is dataset.iml_dataset

    instruction_name:
        type: MLApplication
        dataset: dataset
        config_path: optimal_ml_model.zip # the name of the ML model
        label: disease # the label
        pool_size: 1000
        store_encoded_data: False


Tool output
---------------------------------------------
This Galaxy tool will produce the following history elements:

- ML Model Application Archive: a .zip file containing the complete output folder as it was produced by immuneML. This folder
  contains the output of the MLApplication instruction including ....
  Furthermore, the folder contains the complete YAML specification file for the immuneML run, the HTML output and a log file.

- Results of ML application: a HTML page that allows you to browse through all results, including ...
