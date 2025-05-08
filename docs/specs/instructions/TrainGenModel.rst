



TrainGenModel instruction implements training generative AIRR models on receptor level. Models that can be trained
for sequence generation are listed under Generative Models section.

This instruction takes a dataset as input which will be used to train a model, the model itself, and the number of
sequences to generate to illustrate the applicability of the model. It can also produce reports of the fitted model
and reports of original and generated sequences.

To use the generative model previously trained with immuneML, see :ref:`ApplyGenModel` instruction.


**Specification arguments:**

- dataset: dataset to use for fitting the generative model; it has to be defined under definitions/datasets

- method: which model to fit (defined previously under definitions/ml_methods)

- number_of_processes (int): how many processes to use for fitting the model

- gen_examples_count (int): how many examples (sequences, repertoires) to generate from the fitted model

- reports (list): list of report ids (defined under definitions/reports) to apply after fitting a generative model
  and generating gen_examples_count examples; these can be data reports (to be run on generated examples), ML
  reports (to be run on the fitted model)

- training_percentage (float): percentage of the dataset to use for training the generative model. If set to 1, the
  full dataset will be used for training and the test dataset will be the same as the training dataset. Default
  value is 0.7. When export_combined_dataset is set to True, the splitting of sequences into train, test, and
  generated will be shown in column dataset_split.

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    instructions:
        my_train_gen_model_inst: # user-defined instruction name
            type: TrainGenModel
            dataset: d1 # defined previously under definitions/datasets
            method: model1 # defined previously under definitions/ml_methods
            gen_examples_count: 100
            number_of_processes: 4
            training_percentage: 0.7
            export_generated_dataset: True
            export_combined_dataset: False
            reports: [data_rep1, ml_rep2]


