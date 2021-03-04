How to add a new machine learning method
==========================================

In this tutorial, we will add a new machine learning method. This tutorial assumes you have installed immuneML for development as described at :ref:`Set up immuneML for development`.

To add a new ML method to immuneML, add a class that inherits :py:obj:`~immuneML.ml_methods.MLMethod.MLMethod` class to the :py:mod:`immuneML.ml_methods` package
and implement abstract methods. The name of the new class has to be different from the ML methods’ classes already defined in the same package.


Adding the new method to immuneML
-----------------------------------

For methods based on scikit-learn, read how to do this under :ref:`Adding a method based on scikit-learn`.
For other methods, see :ref:`Adding native methods`.

The ML models in the immuneML support one label. If multiple labels are specified e.g., in :ref:`TrainMLModel` instruction, a separate model will
be fitted to each label.

.. include:: ./dev_docs_util.rst

Adding a method based on scikit-learn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To add a method from the scikit-learn’s package, go through this step-by-step guide where, exemplary, a SVM class based on scikit-learn’s LinearSVC
will be added:

  #. Add a new class to the package ml_methods.

  #. Make SklearnMethod a base class of the new class.

  #. Implement a constructor (handle parameters and parameter_grid as inputs).

  #. Implement get_ml_model(cores_for_training: int) function, which should return a new instance of the desired scikit-learn’s class with the parameters that were passed to the constructor of the new class.

  #. Implement _can_predict_proba() to return True or False to indicate whether the method can output the class probabilities.

  #. Implement get_params(label) to return the coefficients and/or other trained parameters of the model for the given label.

  #. Add class documentation describing how to use the new ML method.

Example scikit-learn-based SVM implementation:

.. code-block:: python

  from sklearn.model_selection import RandomizedSearchCV
  from sklearn.svm import LinearSVC

  from immuneML.ml_methods.SklearnMethod import SklearnMethod


  class MySVM(SklearnMethod):
    """
    This is a wrapper of scikit-learn’s LinearSVC class. Please see the
    `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html>`_
    of LinearSVC for the parameters.

    Note: if you are interested in plotting the coefficients of the SVM model,
    consider running the :ref:`Coefficients` report.

    For usage instructions, check :py:obj:`~immuneML.ml_methods.SklearnMethod.SklearnMethod`.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_svm: # user-defined method name
            SVM: # name of the ML method
                # sklearn parameters (same names as in original sklearn class)
                penalty: l1 # always use penalty l1
                C: [0.01, 0.1, 1, 10, 100] # find the optimal value for C
                # Additional parameter that determines whether to print convergence warnings
                show_warnings: True
            # if any of the parameters under SVM is a list and model_selection_cv is True,
            # a grid search will be done over the given parameters, using the number of folds specified in model_selection_n_folds,
            # and the optimal model will be selected
            model_selection_cv: True
            model_selection_n_folds: 5
        # alternative way to define ML method with default values:
        my_default_svm: SVM

    """

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameters = parameters if parameters is not None else {"max_iter": 10000, "multi_class": "crammer_singer"}
        _parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(SVM, self).__init__(parameter_grid=_parameter_grid, parameters=_parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        return LinearSVC(**self._parameters)

    def can_predict_proba(self) -> bool:
        return False

    def get_params(self):
        params = self.model.get_params()
        params["coefficients"] = self.model.coef_[0].tolist()
        params["intercept"] = self.model.intercept_.tolist()
        return params


Adding native methods
^^^^^^^^^^^^^^^^^^^^^^^

To add a new ML method:

  #. Add a new class in a new file, where the class name and the file name must match.

  #. Make the new class inherit :py:obj:`~immuneML.ml_methods.MLMethod.MLMethod` class.

  #. Define an init function. The constructor arguments in the new class will be the required parameters in the specification file.

  #. Implement all abstract methods as defined in :py:obj:`~immuneML.ml_methods.MLMethod.MLMethod` class.

  #. Add class documentation describing how to use the new ML method.

Testing the new ML method
----------------------------

Add a unit test for the new ML method:

1. Add a new file to :py:mod:`~test.ml_methods` package named test_mySVM.py.
2. Add a class TestMySVM that inherits :code:`unittest.TestCase`.
3. Add a function to set up cache used for testing
4. Define tests for functions you implemented.

The full test example for MySVM class is given below:

.. code-block:: python

  import os
  import pickle
  import shutil
  from unittest import TestCase

  import numpy as np
  from sklearn.svm import LinearSVC

  from immuneML.caching.CacheType import CacheType
  from immuneML.data_model.encoded_data.EncodedData import EncodedData
  from immuneML.environment.Constants import Constants
  from immuneML.environment.EnvironmentSettings import EnvironmentSettings
  from immuneML.ml_methods.MySVM import MySVM # newly added method
  from immuneML.util.PathBuilder import PathBuilder


  class TestSVM(TestCase):

      def setUp(self) -> None:
          os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name # set up cache, always the same

      def test_fit(self):
          x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
          y = {"default": np.array([1, 0, 2, 0])}

          svm = MySVM()
          svm.fit(EncodedData(x, y), 'default') # just test if nothing breaks

      def test_predict(self):
          x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]]) # encoded data, where one row is one example, and columns are features
          y = {"test_label_name": np.array([1, 0, 2, 0])} # classes for the given label, for each example in the dataset

          svm = MySVM() # create an instance of class for testing
          svm.fit(EncodedData(x, y), "test_label_name") # fit the classifier using EncodedData object with includes encoded data and classes for each of the examples

          test_x = np.array([[0, 1, 0], [1, 0, 0]]) # new encoded data for testing the method
          y = svm.predict(EncodedData(test_x), 'test_label_name')["test_label_name"] # extract predictions for new encoded data for the given label

          self.assertTrue(len(y) == 2) # check the number of predictions (2 because there were 2 examples in the new encoded data)
          self.assertTrue(y[0] in [0, 1, 2]) # check that classes are in the list of valid classes for each prediction
          self.assertTrue(y[1] in [0, 1, 2])

      def test_fit_by_cross_validation(self):
          x = EncodedData(np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]]),
                          {"t1": [1, 0, 2, 0, 1, 0, 2, 0], "t2": [1, 0, 2, 0, 1, 0, 2, 0]})

          svm = MySVM()
          svm.fit_by_cross_validation(x, number_of_splits=2, label_name="t1") # check if nothing fails

      def test_store(self):
          x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
          y = {"default": np.array(['a', "b", "c", "a"])}

          svm = MySVM()
          svm.fit(EncodedData(x, y), 'default')

          path = EnvironmentSettings.tmp_test_path / "my_svm_store/"

          svm.store(path)

          # when the trained method is stored, check if the format is as defined in store()
          self.assertTrue(os.path.isfile(path / "svm.pickle"))

          with open(path / "svm.pickle", "rb") as file:
              svm2 = pickle.load(file)

          self.assertTrue(isinstance(svm2, LinearSVC))

          shutil.rmtree(path)

      def test_load(self):
          x = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
          y = {"default": np.array([1, 0, 2, 0])}

          svm = MySVM()
          svm.fit(EncodedData(x, y), 'default')

          path = EnvironmentSettings.tmp_test_path / "my_svm_load/"
          PathBuilder.build(path)

          with open(path / "svm.pickle", "wb") as file:
              pickle.dump(svm.get_model(), file)

          svm2 = MySVM()
          svm2.load(path)

          # when the model is loaded from disk, check if the class matches
          self.assertTrue(isinstance(svm2.get_model(), LinearSVC))

          # optionally, more checks can be added

          shutil.rmtree(path)


Adding a new ML method: additional information
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
      MySVM: # class name of the method
        penalty: l1 # parameters of the model
      model_selection_cv: False # should there be a grid search and cross-validation - not here
      model_selection_n_folds: -1 # no number of folds for cross-validation as it is not used here
    my_svm_cv: # the name of the next method
      MySVM: # class name of the method
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

Full specification that trains the added ML method on the simulated data would look like this:

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
          MySVM: # class name of the method
            C: 10 # parameters of the model
          model_selection_cv: False # should there be a grid search and cross-validation - not here
          model_selection_n_folds: -1 # no number of folds for cross-validation as it is not used here
        my_svm_cv: # the name of the next method
          MySVM: # class name of the method
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