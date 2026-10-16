How to add a new preprocessing
==========================================


Preprocessings are applied to modify a dataset before encoding the data, for example, removing certain sequences from a repertoire.
In immuneML, the sequence of preprocessing steps applied to a given dataset before training an ML model is
considered a hyperparameter that can be optimized using nested cross validation.


Adding an example preprocessor to the immuneML codebase
-------------------------------------------------------------
This tutorial describes how to add a new  :py:obj:`~immuneML.preprocessing.Preprocessor.Preprocessor` class to immuneML,
using a simple example preprocessor. We highly recommend completing this tutorial to get a better understanding of the immuneML
interfaces before continuing to :ref:`implement your own preprocessor <Implementing a new preprocessor>`.


Step-by-step tutorial
^^^^^^^^^^^^^^^^^^^^^^^^^

For this tutorial, we provide a :code:`SillyFilter` (:download:`download here <./example_code/SillyFilter.py>` or view below),
in order to test adding a new Preprocessor file to immuneML. This preprocessor acts like a filter which randomly selects
a subset of repertoires to keep.

    .. collapse:: SillyFilter.py

      .. literalinclude:: ./example_code/SillyFilter.py
         :language: python


#. Add a new class to the :py:mod:`~immuneML.preprocessing.filters` package inside the :py:mod:`~immuneML.preprocessing` package.
   The new class should inherit from the base class :py:obj:`~immuneML.preprocessing.filters.Filter.Filter`.
   A filter is a special category of preprocessors which removes examples (repertoires) from the dataset.
   Other preprocessors, which for example just annotate the dataset, should be placed directly inside the :py:mod:`~immuneML.preprocessing` package
   and inherit the :py:mod:`~immuneML.preprocessing.Preprocessor.Preprocessor` class instead.

#. If the preprocessor has any default parameters, they should be added in a default parameters YAML file. This file should be added to the folder :code:`config/default_params/preprocessing`.
   The default parameters file is automatically discovered based on the name of the class using the base name converted to snake case, and with an added '_params.yaml' suffix.
   For the :code:`SillyFilter`, this is :code:`silly_filter_params.yaml`, which could for example contain the following:

   .. code:: yaml

      fraction_to_keep: 0.8

   In rare cases where classes have unconventional names that do not translate well to CamelCase (e.g., MiXCR, VDJdb), this needs to be accounted for in :py:meth:`~immuneML.dsl.DefaultParamsLoader.convert_to_snake_case`.


Test running the new preprocessing with a YAML specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to use immuneML directly to test run your preprocessor, the YAML example below may be used.
This example analysis creates a randomly generated dataset, runs the :code:`SillyFilter`, and
runs the :ref:`SimpleDatasetOverview` report on the preprocessed dataset to inspect the results.

   .. collapse:: test_run_silly_filter.yaml

      .. code:: yaml

         definitions:
           datasets:
             my_dataset:
               format: RandomSequenceDataset
               params:
                 sequence_count: 100

           preprocessing_sequences:
             my_preprocessing:
             - step1:
                 SillyFilter:
                   fraction_to_remove: 0.8

           reports:
             simple_overview: SimpleDatasetOverview


         instructions:
           exploratory_instr:
             type: ExploratoryAnalysis
             analyses:
               analysis_1:
                 dataset: d1
                 preprocessing_sequence: my_preprocessing_seq
                 report: simple_overview





Adding a Unit test for a Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add a unit test for the new :code:`SillyFilter` (:download:`download <./example_code/_test_sillyFilter.py>` the example testfile or view below)

    .. collapse:: test_sillyFilter.py

      .. literalinclude:: ./example_code/_test_sillyFilter.py
         :language: python


#. Add a new file to the :code:`test.preprocessing.filters` package named test_sillyFilter.py.
#. Add a class :code:`TestSillyFilter` that inherits :code:`unittest.TestCase` to the new file.
#. Add a function :code:`setUp()` to set up cache used for testing. This should ensure that the cache location will be set to :code:`EnvironmentSettings.tmp_test_path / "cache/"`
#. Define one or more tests for the class and functions you implemented.

   - It is recommended to at least test building the Preprocessor and running the preprocessing
   - Mock data is typically used to test new classes. Tip: the :code:`RandomDatasetGenerator` class can be used to generate Repertoire, Sequence or Receptor datasets with random sequences.
   - If you need to write data to a path (for example test datasets or results), use the following location: :code:`EnvironmentSettings.tmp_test_path / "some_unique_foldername"`



Implementing a new Preprocessor
------------------------------------------

This section describes tips and tricks for implementing your own new :code:`Preprocessor` from scratch.
Detailed instructions of how to implement each method, as well as some special cases, can be found in the
:py:obj:`~immuneML.preprocessing.Preprocessor.Preprocessor` base class.


.. include:: ./coding_conventions_and_tips.rst



Implementing the process() method in a new encoder class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main functionality of the preprocessor class is implemented in its :code:`process(dataset, params)` method.
This method takes in a dataset, modifies the dataset according to the given instructions, and returns
the new modified dataset.


When implementing the :code:`process(dataset, params)` method, take the following points into account:

- The method takes in the argument :code:`params`, which is a dictionary containing any relevant parameters.
  One of these parameters is the result path :code:`params["result_path"]` which should be used as the location
  to store the metadata file of a new repertoire dataset.

- Check if the given dataset is the correct dataset type, for example by using the static method :code:`check_dataset_type(dataset, valid_dataset_types, location)`.
  Some preprocessings are only sensible for a given type of dataset. Datasets can be of the type RepertoireDataset, SequenceDataset and ReceptorDataset (see: :ref:`immuneML data model`).

- Do not modify the given dataset object, but create a clone instead.

- When your preprocessor is a filter (i.e., when it removes sequences or repertoires from the dataset), extra precautions
  need to be taken to ensure that the dataset does not contain empty repertoires and that the entries in the metadata
  file match the new dataset. The utility functions provided by the :py:mod:`~immuneML.preprocessing.filters.Filter.Filter`
  class can be useful for this:

  - :code:`remove_empty_repertoires(repertoires)` checks whether any of the provided repertoires are empty
    (this might happen when filtering out sequences based on strict criteria), and returns a list containing only non-empty repertoires.

  - :code:`check_dataset_not_empty(processed_dataset, location)` checks whether there is still any data left in the dataset.
    If all sequences or repertoires were removed by filtering, an error will be thrown.

  - :code:`build_new_metadata(dataset, indices_to_keep, result_path)` creates a new metadata file based on a subset of the
    existing metadata file. When removing repertoires from a repertoire dataset, this method should always be applied.




Class documentation standards for preprocessors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. include:: ./class_documentation_standards.rst


.. collapse:: Click to view a full example of Preprocessor class documentation.

       .. code::

           This SillyFilter class is a placeholder for a real Preprocessor.
           It randomly selects a fraction of the repertoires to be removed from the dataset.


           **Specification arguments:**

           - fraction_to_keep (float): The fraction of repertoires to keep


           **YAML specification:**

           .. indent with spaces
           .. code-block:: yaml

               definitions:
                   preprocessing_sequences:
                       my_preprocessing:
                           - step1:
                               SillyFilter:
                                   fraction_to_remove: 0.8