How to add a new preprocessing
==========================================

In this tutorial, we will add a new preprocessing to immuneML.
This tutorial assumes you have installed immuneML for development as described at :ref:`Set up immuneML for development`.

Preprocessings are applied to modify a dataset before encoding the data, for example, removing certain sequences from a repertoire.
In immuneML, the sequence of preprocessing steps applied to a given dataset before training an ML model is
considered a hyperparameter that can be optimized using nested cross validation.


Adding a new Preprocessor class
-------------------------------

All preprocessing methods should be placed in the :py:mod:`~immuneML.preprocessing` package, and inherit the immuneML
class :py:mod:`~immuneML.preprocessing.Preprocessor.Preprocessor`.
A filter is a special category of preprocessors which removes sequences or repertoires from the dataset.
If your preprocessing is a filter, it should be placed in the :py:mod:`~immuneML.preprocessing.filters` package and
inherit the :py:mod:`~immuneML.preprocessing.filters.Filter.Filter` class.

A new preprocessor should implement:

- an :code:`__init__()` method if the preprocessor uses any parameters.
- The static abstract method :code:`process(dataset, params)`, which takes a dataset and returns a new (modified) dataset.
- The abstract method :code:`process_dataset(dataset, result_path)`, which typically prepares parameters and calls :code:`process(dataset, params)` internally.

An example implementation of a new filter named NewClonesPerRepertoireFilter is shown below.
It includes implementations of the abstract methods and class documentation at the beginning. This class documentation will be shown to the user.

.. code-block:: python

    class NewClonesPerRepertoireFilter(Filter):
        """
        Removes all repertoires from the RepertoireDataset, which contain fewer clonotypes than specified by the
        lower_limit, or more clonotypes than specified by the upper_limit.
        Note that this filter filters out repertoires, not individual sequences, and can thus only be applied to RepertoireDatasets.

        Arguments:

            lower_limit (int): The minimal inclusive lower limit for the number of clonotypes allowed in a repertoire.

            upper_limit (int): The maximal inclusive upper limit for the number of clonotypes allowed in a repertoire.

        When no lower or upper limit is specified, or the value -1 is specified, the limit is ignored.


        YAML specification:

        .. indent with spaces
        .. code-block:: yaml

            preprocessing_sequences:
                my_preprocessing:
                    - my_filter:
                        NewClonesPerRepertoireFilter:
                            lower_limit: 100
                            upper_limit: 100000

        """

        def __init__(self, lower_limit: int = -1, upper_limit: int = -1):
            self.lower_limit = lower_limit
            self.upper_limit = upper_limit

        def process_dataset(self, dataset: RepertoireDataset, result_path: Path = None):
            # Prepare the parameter dictionary for the process method
            params = {"result_path": result_path}
            if self.lower_limit > -1:
                params["lower_limit"] = self.lower_limit
            if self.upper_limit > -1:
                params["upper_limit"] = self.upper_limit

            return NewClonesPerRepertoireFilter.process(dataset, params)

        @staticmethod
        def process(dataset: RepertoireDataset, params: dict) -> RepertoireDataset:
            # Check if the dataset is the correct type for this preprocessor (here, only RepertoireDataset is allowed)
            NewClonesPerRepertoireFilter.check_dataset_type(dataset, [RepertoireDataset], "NewClonesPerRepertoireFilter")

            processed_dataset = dataset.clone()

            # Here, any code can be placed to create a modified set of repertoires
            repertoires = []
            indices = []
            for index, repertoire in enumerate(dataset.get_data()):
                if "lower_limit" in params.keys() and len(repertoire.sequences) >= params["lower_limit"] or \
                    "upper_limit" in params.keys() and len(repertoire.sequences) <= params["upper_limit"]:
                    repertoires.append(dataset.repertoires[index])
                    indices.append(index)

            processed_dataset.repertoires = repertoires

            # Rebuild the metadata file, which only contains the repertoires that were kept
            processed_dataset.metadata_file = NewClonesPerRepertoireFilter.build_new_metadata(dataset, indices, params["result_path"])

            # Ensure the dataset did not end up empty after filtering
            NewClonesPerRepertoireFilter.check_dataset_not_empty(processed_dataset, "NewClonesPerRepertoireFilter")

            return processed_dataset

Unit testing the new Preprocessor
---------------------------------

To add a unit test:

#. Create a new python file named test_newClonesPerRepertoireFilter.py and add it to the :py:mod:`~test.preprocessing.filters` test package.
#. Add a class TestNewClonesPerRepertoireFilter that inherits :code:`unittest.TestCase` to the new file.
#. Add a function :code:`setUp()` to set up cache used for testing (see example below).
#. Define one or more tests for the class and functions you implemented.
#. If you need to write data to a path (for example test datasets or results), use the following location: :code:`EnvironmentSettings.root_path / "/test/tmp/some_unique_foldername"`

When building unit tests, a useful class is :py:obj:`~immuneML.simulation.dataset_generation.RandomDatasetGenerator.RandomDatasetGenerator`, which can create a dataset with random sequences.

An example of the unit test TestNewClonesPerRepertoireFilter is given below.

.. code-block:: python

    import os
    import shutil
    from unittest import TestCase

    from immuneML.caching.CacheType import CacheType
    from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
    from immuneML.environment.Constants import Constants
    from immuneML.environment.EnvironmentSettings import EnvironmentSettings
    from immuneML.preprocessing.filters.ClonesPerRepertoireFilter import ClonesPerRepertoireFilter
    from immuneML.util.PathBuilder import PathBuilder
    from immuneML.util.RepertoireBuilder import RepertoireBuilder


    class TestClonesPerRepertoireFilter(TestCase):

        def setUp(self) -> None:
            os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

        def test_process(self):
            path = EnvironmentSettings.root_path / "test/tmp/clones_per_repertoire_filter/"
            PathBuilder.build(path)
            dataset = RepertoireDataset(repertoires=RepertoireBuilder.build([["ACF", "ACF", "ACF"],
                                                                           ["ACF", "ACF"],
                                                                           ["ACF", "ACF", "ACF", "ACF"]], path)[0])

            dataset1 = ClonesPerRepertoireFilter.process(dataset, {"lower_limit": 3, "result_path": path})
            self.assertEqual(2, dataset1.get_example_count())

            dataset2 = ClonesPerRepertoireFilter.process(dataset, {"upper_limit": 2, "result_path": path})
            self.assertEqual(1, dataset2.get_example_count())

            self.assertRaises(AssertionError, ClonesPerRepertoireFilter.process, dataset, {"lower_limit": 10, "result_path": path})

            shutil.rmtree(path)




Adding a Preprocessor: additional information
------------------------------------------


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
  Some preprocessings are only sensible for a given type of dataset. Datasets can be of the type RepertoireDataset, SequenceDataset and ReceptorDataset.

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



Test run of the preprocessing: specifying the preprocessing in YAML
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Custom preprocessings can be specified in the YAML specification just like any other existing preprocessing. The preprocessing
needs to be defined under ‘definitions’ and referenced in the ‘instructions’ section.
The easiest way to test a new preprocessing is to apply it when running the :ref:`ExploratoryAnalysis` instruction,
and running the :ref:`SimpleDatasetOverview` report on the preprocessed dataset to inspect the results.
An example YAML specification for this is given below:

.. code-block:: yaml

  definitions:
    preprocessing_sequences:
      my_preprocessing_seq:           # User-defined name of the preprocessing sequence (may contain one or more preprocessings)
      - my_new_filter:                # User-defined name of one preprocessing
        NewClonesPerRepertoireFilter: # The name of the new preprocessor class
          lower_limit: 10             # Any parameters to provide to the preprocessor.
          upper_limit: 20             # In this test example, only repertoires with 10-20 clones are kept

    datasets:
      d1:
        # if you do not have real data to test your report with, consider
        # using a randomly generated dataset, see the documentation:
        # “How to generate a random receptor or repertoire dataset”
        format: RandomRepertoireDataset
        params:
            repertoire_count: 100 # number of random repertoires to generate
            sequence_count_probabilities:
              15: 0.5  # Generate a dataset where half the repertoires contain 15 sequences, and the other half 25 sequences
              25: 0.5  # When the filter is applied, only the 50 repertoires with 15 sequences should remain

    reports:
      simple_overview: SimpleDatasetOverview

  instructions:
    exploratory_instr: # Example of specifying reports in ExploratoryAnalysis
      type: ExploratoryAnalysis
      analyses:
        analysis_1: # Example analysis with data report
          dataset: d1
          preprocessing_sequence: my_preprocessing_seq # apply the preprocessing
          report: simple_overview



Adding class documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To complete the preprocessing, class documentation should be added to inform other users how the preprocessing should be used.
The documentation should contain:

  #. A short, general description of the preprocessor, including which dataset types (repertoire dataset, sequence dataset, receptor dataset) it can be applied to.

  #. If applicable, a listing of the types and descriptions of the arguments that should be providedto the preprocessor.

  #. An example of how the preprocessor definition should be specified in the YAML.

The class docstrings are used to automatically generate the documentation for the preprocessor, and should be written
in Sphinx `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ formatting.

This is the example of documentation for :py:obj:`~immuneML.preprocessing.filters.ClonesPerRepertoireFilter.ClonesPerRepertoireFilter`:

.. code-block:: RST

    Removes all repertoires from the RepertoireDataset, which contain fewer clonotypes than specified by the
    lower_limit, or more clonotypes than specified by the upper_limit.
    Note that this filter filters out repertoires, not individual sequences, and can thus only be applied to RepertoireDatasets.

    Arguments:

        lower_limit (int): The minimal inclusive lower limit for the number of clonotypes allowed in a repertoire.

        upper_limit (int): The maximal inclusive upper limit for the number of clonotypes allowed in a repertoire.
        When no lower or upper limit is specified, or the value -1 is specified, the limit is ignored.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        preprocessing_sequences:
            my_preprocessing:
                - my_filter:
                    NewClonesPerRepertoireFilter:
                        lower_limit: 100
                        upper_limit: 100000