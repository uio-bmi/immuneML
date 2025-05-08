


DatasetExport instruction takes a list of datasets as input, optionally applies preprocessing steps, and outputs
the data in specified formats.

**Specification arguments:**

- datasets (list): a list of datasets to export in all given formats

- preprocessing_sequence (str): which preprocessing sequence to use on the dataset(s), this item is optional and does not have to be specified.
  When specified, the same preprocessing sequence will be applied to all datasets.

- formats (list): a list of formats in which to export the datasets. Valid formats are class names of any
  non-abstract class inheriting :py:obj:`~immuneML.IO.dataset_export.DataExporter.DataExporter`.

- number_of_processes (int): how many processes to use during repertoire export (not used for sequence datasets)

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    instructions:
        my_dataset_export_instruction: # user-defined instruction name
            type: DatasetExport # which instruction to execute
            datasets: # list of datasets to export
                - my_generated_dataset
                - my_dataset_from_adaptive
            preprocessing_sequence: my_preprocessing_sequence
            number_of_processes: 4
            export_formats: # list of formats to export the datasets to
                - AIRR
                - ImmuneML


