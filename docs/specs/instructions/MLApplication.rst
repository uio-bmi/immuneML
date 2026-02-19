


Instruction which enables using trained ML models and encoders on new datasets which do not necessarily have labeled data.
When the same label is provided as the ML setting was trained for, performance metrics can be computed.

The predictions are stored in the predictions.csv in the result path in the following format:

.. list-table::
    :widths: 25 25 25 25
    :header-rows: 1

    * - example_id
      - cmv_predicted_class
      - cmv_1_proba
      - cmv_0_proba
    * - e1
      - 1
      - 0.8
      - 0.2
    * - e2
      - 0
      - 0.2
      - 0.8
    * - e3
      - 1
      - 0.78
      - 0.22


If the same label that the ML setting was trained for is present in the provided dataset, the 'true' label value
will be added to the predictions table in addition:

.. list-table::
    :widths: 20 20 20 20 20
    :header-rows: 1

    * - example_id
      - cmv_predicted_class
      - cmv_1_proba
      - cmv_0_proba
      - cmv_true_class
    * - e1
      - 1
      - 0.8
      - 0.2
      - 1
    * - e2
      - 0
      - 0.2
      - 0.8
      - 0
    * - e3
      - 1
      - 0.78
      - 0.22
      - 0

**Specification arguments:**

- dataset: dataset for which examples need to be classified

- config_path: path to the zip file exported from MLModelTraining instruction (which includes train ML model, encoder, preprocessing etc.)

- number_of_processes (int): how many processes should be created at once to speed up the analysis. For personal machines, 4 or 8 is usually a good choice.

- metrics (list): a list of metrics (`accuracy`, `balanced_accuracy`, `confusion_matrix`, `f1_micro`, `f1_macro`, `f1_weighted`, `precision`, `precision_micro`, `precision_macro`, `precision_weighted`, `recall_micro`, `recall_macro`, `recall_weighted`, `average_precision`, `brier_score`, `recall`, `auc`, `auc_ovo`, `auc_ovr`, `log_loss`, `specificity`) to compute between the true and predicted classes. These metrics will only be computed when the same label with the same classes is provided for the dataset as the original label the ML setting was trained for.


**YAML specification:**

.. highlight:: yaml
.. code-block:: yaml

    instructions:
        instruction_name:
            type: MLApplication
            dataset: d1
            config_path: ./config.zip
            metrics:
            - accuracy
            - precision
            - recall
            number_of_processes: 4


