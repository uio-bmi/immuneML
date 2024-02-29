How to add a new machine learning method
==========================================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML dev docs: add a new machine learning method
   :twitter:description: See how to add a new machine learning method to the immuneML platform.
   :twitter:image: https://docs.immuneml.uio.no/_images/extending_immuneML.png

In this tutorial, we will add a new machine learning method. This tutorial assumes you have installed immuneML for development as described at :ref:`Set up immuneML for development`.

To add a new ML method to immuneML, add a class that inherits :py:obj:`~immuneML.ml_methods.MLMethod.MLMethod` class to the :py:mod:`immuneML.ml_methods` package
and implement abstract methods. The name of the new class has to be different from the ML methods’ classes already defined in the same package.

.. note::

  The initial development of the new ML method need not take place within immuneML. immuneML can be used to prepare, encode and export the data for
  developing the method using the :ref:`ExploratoryAnalysis` instruction, desired encoding, and :ref:`DesignMatrixExporter` report. For more details,
  see :ref:`Testing the ML method outside immuneML with a sample design matrix`. The method can
  then be developed and debugged separately, and later integrated into the platform as described below to fully benefit from available immuneML
  functionalities related to importing datasets from different formats, using various data representations, benchmarking against existing methods and
  robustly assessing the performance.


Adding a new MLMethod class
-----------------------------------


in :ref:`TrainMLModel` instruction, a separate model will be fitted to each label.

.. include:: ./dev_docs_util.rst

To add a new ML method:

  #. Add a new class in a new file, where the class name and the file name must match.

  #. Make the new class inherit :py:obj:`~immuneML.ml_methods.MLMethod.MLMethod` class.

  #. Define an init function. The constructor arguments in the new class will be the required parameters in the specification file.

  #. Implement all abstract methods as defined in :py:obj:`~immuneML.ml_methods.MLMethod.MLMethod` class.

  #. Add class documentation describing how to use the new ML method.

<todo add default parameters>

<todo ecample code>

Special case: adding a method based on scikit-learn
-----------------------------------------------------



To add a method from the scikit-learn’s package, go through this step-by-step guide where, exemplary, a SVM class based on scikit-learn’s LinearSVC
will be added:

  #. Add a new class to the package ml_methods.

  #. Make SklearnMethod a base class of the new class.

  #. Implement a constructor (handle parameters and parameter_grid as inputs).

  #. Implement get_ml_model(cores_for_training: int) function, which should return a new instance of the desired scikit-learn’s class with the parameters that were passed to the constructor of the new class.

  #. Implement _can_predict_proba() to return True or False to indicate whether the method can output the class probabilities.

  #. Implement get_params(label) to return the coefficients and/or other trained parameters of the model for the given label.

  #. Add class documentation describing how to use the new ML method.

<todo add default params>

- class naming
- name of defaultparams file

<todo ecample code>

Adding a Unit test for the MLMethod
------------------------------------

Add a unit test for the new ML method:

#. Add a new file to :py:mod:`~test.ml_methods` package named test_newSVM.py.
#. Add a class TestNewSVM that inherits :code:`unittest.TestCase` to the new file..
#. Add a function :code:`setUp()` to set up cache used for testing (see example below). This will ensure that the cache location will be set to :code:`EnvironmentSettings.tmp_test_path / "cache/"`
#. Define one or more tests for the class and functions you implemented.
#. If you need to write data to a path (for example test datasets or results), use the following location: :code:`EnvironmentSettings.tmp_test_path / "some_unique_foldername"`

When building unit tests, a useful class is :py:obj:`~immuneML.simulation.dataset_generation.RandomDatasetGenerator.RandomDatasetGenerator`, which can create a dataset with random sequences.



Adding a new MLMethod: additional information
-----------------------------------------------

To test the method outside immuneML, see :ref:`Testing the ML method outside immuneML with a sample design matrix`.

Using ML methods from specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use ML method from specification, it is necessary to define:

  #. The method class,
  #. Parameters for the method,
  #. And if applicable, whether cross-validation should be performed to determine the optimal parameters.

The cross-validation performs the grid search over the parameters if any of the parameters is specified as a list of potential values.

An example specification for support vector machine without cross-validation (my_svm), and support vector machine with cross-validation (my_svm_cv) would be:

.. indent with spaces
.. code-block:: yaml

  ml_methods:
    my_svm: # the name of the method which will be used in the specification to refer to the method
      NewSVM: # class name of the method
        penalty: l1 # parameters of the model
      model_selection_cv: False # should there be a grid search and cross-validation - not here
      model_selection_n_folds: -1 # no number of folds for cross-validation as it is not used here
    my_svm_cv: # the name of the next method
      NewSVM: # class name of the method
        penalty:	# parameter of the model
          - l1 # value of the parameter to test
          - l2 # another value of the parameter to test
      model_selection_cv: True # perform cross-validation and grid search
      model_selection_n_folds: 5 # do 5-fold cross-validation

The parameters model_selection_cv and model_selection_n_folds have values False and -1, respectively and can be omitted if there should be no model
selection on this level. Also, if no parameters of the model are specified (such as penalty in the example), default values would be used.

During parsing, the parameters of the model will be assigned to “parameters” attribute of the ML method object if none of the parameters is a list of
possible values. Otherwise, the parameters will be assigned to the parameter_grid parameter which will be later used for grid search and
cross-validation.

Full specification that simulates the data and trains the added ML method on that data would look like this:

.. indent with spaces
.. code-block:: yaml

    definitions:
      datasets:
        my_simulated_data:
          format: RandomRepertoireDataset
          params:
            repertoire_count: 50 # a dataset with 50 repertoires
            sequence_count_probabilities: # each repertoire has 10 sequences
              10: 1
            sequence_length_probabilities: # each sequence has length 15
              15: 1
            labels:
              my_label: # half of the repertoires has my_label = true, the rest has false
                false: 0.5
                true: 0.5
      encodings:
        my_3mer_encoding:
          KmerFrequency:
            k: 3
      ml_methods:
        my_svm: # the name of the method which will be used in the specification to refer to the method
          NewSVM: # class name of the method
            C: 10 # parameters of the model
          model_selection_cv: False # should there be a grid search and cross-validation - not here
          model_selection_n_folds: -1 # no number of folds for cross-validation as it is not used here
        my_svm_cv: # the name of the next method
          NewSVM: # class name of the method
            penalty:	# parameter of the model
              - l1 # value of the parameter to test
              - l2 # another value of the parameter to test
          model_selection_cv: True # perform cross-validation and grid search
          model_selection_n_folds: 5 # do 5-fold cross-validation
    instructions:
      train_new_ml_methods_inst:
        type: TrainMLModel
        dataset: my_simulated_data
        assessment: # outer cross-validation loop splitting the data into training and test datasets
          split_strategy: random
          split_count: 1
          training_percentage: 0.7
        selection: # inner cross-validation loop splitting the training data into training and validation datasets
          split_strategy: random
          split_count: 1
          training_percentage: 0.7
        labels: [my_label]
        optimization_metric: balanced_accuracy
        settings:
          - encoding: my_3mer_encoding
            ml_method: my_svm
          - encoding: my_3mer_encoding
            ml_method: my_svm_cv # this method will do a third level of cross-validation to select optimal penalty as described above

To run this from the root directory of the project, save the specification to specs.yaml and run the following:

.. code-block:: console

  immune-ml specs.yaml output_dir/

Compatible encoders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each ML method is only compatible with a limited set of encoders. immuneML automatically checks if the given encoder and ML method are
compatible when running the TrainMLModel instruction, and raises an error if they are not compatible.
To ensure immuneML recognizes the encoder-ML method compatibility, make sure that the encoder(s) of interest is added to the list
of encoder classes returned by the :code:`get_compatible_encoders()` method of the ML method.