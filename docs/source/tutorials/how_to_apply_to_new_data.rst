How to apply previously trained ML models to a new dataset
=========================================================================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML: apply trained ML models to a new dataset
   :twitter:description: See tutorials on how to apply previously trained ML models to new datasets.
   :twitter:image: https://docs.immuneml.uio.no/_images/receptor_classification_overview.png


When you train an ML model to classify a label on a given dataset using the :ref:`TrainMLModel` instruction,
the optimal ML settings (a trained model, encoding, and optionally preprocessing) for each label are exported.
These ML setting configurations can subsequently be used to predict that same label on a new dataset
for which the true labels are not known. This is done using the :ref:`MLApplication` instruction.
This instruction will output a table with the predictions on the new dataset, and the probabilities
that these predictions were based on.

Note that the exported ML settings include encodings and preprocessing steps, which are considered hyperparameters
inside immuneML. It is thus not possible to apply the ML model on data that was encoded or preprocessed in a different
way.

One should also be aware that the way the data was generated or preprocessed before being used inside immuneML
can also have an effect on the results. In other words, if there are major differences in how the datasets were
generated, for example if the data was sequenced using a different platform, then the predictions of the ML model
may not be as correct on the new dataset as they were on the original test dataset.

For a tutorial on training ML models, see: :ref:`How to train and assess a receptor or repertoire-level ML classifier`

For a tutorial on importing datasets to immuneML (for training or applying an ML model on the dataset), see :ref:`How to import data into immuneML`.

YAML specification example using the MLApplication instruction
------------------------------------------------------------------
The :ref:`MLApplication` instruction takes in a :code:`dataset` and a :code:`config_path`. The :code:`config_path` should
point at one of the .zip files exported by the previously run :ref:`TrainMLModel` instruction. They can be found in the sub-folder
:code:`instruction_name/optimal_label_name` in the results folder.


.. highlight:: yaml
.. code-block:: yaml

  definitions:
    datasets:
      # imported dataset on which the ML model will be applied
      my_dataset: # user-defined dataset name
        format: AIRR
        params:
          metadata_file: path/to/metadata.csv
          path: path/to/data/

  instructions:
    instruction_name:
      type: MLApplication
      dataset: my_dataset
      # path to the exported optimal ML settings file for label 'disease'
      config_path: results/instruction_name/optimal_disease/zip/ml_settings_disease.zip
      number_of_processes: 4
      store_encoded_data: False