How to add a new machine learning method
==========================================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML dev docs: add a new machine learning method
   :twitter:description: See how to add a new machine learning method to the immuneML platform.
   :twitter:image: https://docs.immuneml.uio.no/_images/extending_immuneML.png


Adding an example classifier to the immuneML codebase
-----------------------------------------------------------


This tutorial describes how to add a new  :py:obj:`~immuneML.ml_methods.classifiers.MLMethod.MLMethod` class to immuneML,
using a simple example classifier. We highly recommend completing this tutorial to get a better understanding of the immuneML
interfaces before continuing to :ref:`implement your own classifier <Implementing a new classifier>`.



Step-by-step tutorial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For this tutorial, we provide a :code:`SillyClassifier` (:download:`download here <./example_code/SillyClassifier.py>` or view below), in order to test adding a new :code:`MLMethod` file to immuneML.
This method ignores the input dataset, and makes a random prediction per example.

        .. collapse:: SillyClassifier.py

          .. literalinclude:: ./example_code/SillyClassifier.py
             :language: python


#. Add a new class to the :py:mod:`immuneML.ml_methods.classifiers` package.
   The new class should inherit from the base class :py:obj:`~immuneML.ml_methods.MLMethod.MLMethod`.

#. If the ML method has any default parameters, they should be added in a default parameters YAML file.
   This file should be added to the folder :code:`config/default_params/ml_methods`.
   The default parameters file is automatically discovered based on the name of the class using the class name converted to snake case, and with an added '_params.yaml' suffix.
   For the :code:`SillyClassifier`, this is :code:`silly_classifier_params.yaml`, which could for example contain the following:

   .. code:: yaml

      random_seed: 1

   In rare cases where classes have unconventional names that do not translate well to CamelCase (e.g., MiXCR, VDJdb), this needs to be accounted for in :py:meth:`~immuneML.dsl.DefaultParamsLoader.convert_to_snake_case`.

#. **Use the automated script** `check_new_ml_method.py <https://github.com/uio-bmi/immuneML/blob/master/scripts/check_new_ml_method.py>`_ **to test the newly added ML method.**
   This script will throw errors or warnings if the MLMethod class implementation is incorrect.

   - Note: this script will try running the new classifier with a random :code:`EncodedData` object (a matrix of random numbers), which may not be compatible with your particular MLMethod.
     You may overwrite the function :code:`get_example_encoded_data()` to supply a custom EncodedData object which meets the requirements of your MLMethod.

   Example command to test the :code:`SillyClassifier`:

   .. code:: bash

      python3 ./scripts/check_new_ml_method.py -m ./immuneML/ml_methods/classifiers/SillyClassifier.py

Test running the new ML method with a YAML specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to use immuneML directly to test run your ML method, the YAML example below may be used.
This example analysis encodes a random dataset using k-mer encoding, trains and compares the performance of two silly
classifiers which were initialised with different random seeds, and shows the results in a report.
Note that when you test your own classifier, a compatible encoding must be used.

           .. collapse:: test_run_silly_classifier.yaml

              .. code:: yaml

                 definitions:
                   datasets:
                     my_dataset:
                       format: RandomSequenceDataset
                       params:
                         sequence_count: 100
                         labels:
                           binds_epitope:
                             True: 0.6
                             False: 0.4

                   encodings:
                     my_encoding:
                       KmerFrequency:
                         k: 3

                   ml_methods:
                     my_first_silly_classifier:
                       SillyClassifier:
                         random_seed: 1
                     my_second_silly_classifier:
                       SillyClassifier:
                         random_seed: 2

                   reports:
                     my_training_performance: TrainingPerformance
                     my_settings_performance: MLSettingsPerformance

                 instructions:
                   my_instruction:
                     type: TrainMLModel

                     dataset: my_dataset
                     labels:
                     - binds_epitope

                     settings:
                     - encoding: my_encoding
                       ml_method: my_first_silly_classifier
                     - encoding: my_encoding
                       ml_method: my_second_silly_classifier

                     assessment:
                       split_strategy: random
                       split_count: 1
                       training_percentage: 0.7
                       reports:
                         models: [my_training_performance]
                     selection:
                       split_strategy: random
                       split_count: 1
                       training_percentage: 0.7

                     optimization_metric: balanced_accuracy
                     reports: [my_settings_performance]

Adding a Unit test for an MLMethod
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add a unit test for the new :code:`SillyClassifier` (:download:`download <./example_code/_test_sillyClassifier.py>` the example testfile or view below)

        .. collapse:: test_sillyClassifier.py

          .. literalinclude:: ./example_code/_test_sillyClassifier.py
             :language: python


#. Add a new file to the :code:`test.ml_methods` package named test_sillyClassifier.py.
#. Add a class :code:`TestSillyClassifier` that inherits :code:`unittest.TestCase` to the new file.
#. Add a function :code:`setUp()` to set up cache used for testing. This should ensure that the cache location will be set to :code:`EnvironmentSettings.tmp_test_path / "cache/"`
#. Define one or more tests for the class and functions you implemented.

   - It is recommended to at least test fitting, prediction and storing/loading of the model.
   - Mock data is typically used to test new classes.
   - If you need to write data to a path (for example test datasets or results), use the following location: :code:`EnvironmentSettings.tmp_test_path / "some_unique_foldername"`



Implementing a new classifier
------------------------------------

This section describes tips and tricks for implementing your own new :code:`MLMethod` from scratch.
Detailed instructions of how to implement each method, as well as some special cases, can be found in the
:py:obj:`~immuneML.ml_methods.classifiers.MLMethod.MLMethod` base class.


.. include:: ./coding_conventions_and_tips.rst


Developing a method outside immuneML with a sample design matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The initial development of the new ML method need not take place within immuneML.
immuneML can be used to encode and export an example design matrix using the :ref:`DesignMatrixExporter` report
with an appropriate encoding in the :ref:`ExploratoryAnalysis` instruction.
The method can then be developed and debugged separately, and afterwards be integrated into the platform.

The following YAML example shows how to generate some random example data (:ref:`detailed description here <How to generate a dataset with random sequences>`),
encode it using a k-mer encoding and export the design matrix to .csv format.
Note that for design matrices beyond 2 dimensions (such as :code:`OneHotEncoder` with flatten = False), the matrix is exported as a .npy file instead of a .csv file.

        .. collapse:: export_design_matrix.yaml

          .. code-block:: yaml

            definitions:
              datasets:
                my_simulated_data:
                  format: RandomRepertoireDataset
                  params:
                    repertoire_count: 5 # a dataset with 5 repertoires
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
              reports:
                my_design_matrix:
                  DesignMatrixExporter:
                    name: my_design_matrix
            instructions:
              my_instruction:
                type: ExploratoryAnalysis
                analyses:
                  my_analysis:
                    dataset: my_simulated_data
                    encoding: my_3mer_encoding
                    labels:
                    - my_label
                    report: my_design_matrix


The resulting design matrix can be found the sub-folder :code:`my_instruction/analysis_my_analysis/report/design_matrix.csv`,
and the true classes for each repertoire can be found in :code:`labels.csv`.
To load files into an :code:`EncodedData` object, the function :py:obj:`immuneML.dev_util.util.load_encoded_data` can be used.

Input and output for the fit() and predict() methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inside immuneML, the design matrix is passed to an MLMethod wrapped in an :code:`EncodedData` object.
This is the main input to the fitting and prediction methods.
Additional inputs to the MLMethod during fitting are set in :code:`MLMethod._initialize_fit()`.

The :code:`EncodedData` object contains the following fields:

.. include:: ./encoded_data_object.rst

The output predictions should be formatted the same way as the :code:`EncodedData.labels`:

.. code:: python

  {'label_name': ['class1', 'class1', 'class2']}

When predicting probabilities, a nested dictionary should be used to give the probabilities per class:

.. code:: python

  {'label_name': {'class1': [0.9, 0.8, 0.3]},
                 {'class2': [0.1, 0.2, 0.7]}}


Adding encoder compatibility to an ML method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each ML method is only compatible with a limited set of encoders. immuneML automatically checks if the given encoder and ML method are
compatible when running the TrainMLModel instruction, and raises an error if they are not compatible.
To ensure immuneML recognizes the encoder-ML method compatibility, make sure that the encoder is added to the list of encoder classes
returned by the :code:`get_compatible_encoders()` method of the ML method(s) of interest.


Implementing fitting through cross-validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default, models in immuneML are fitted through nested-cross validation.
This allows for both hyperparameter selection and model comparison.
immuneML also allows for the implementation of a third level of k-fold cross-validation for hyperparameter selection within
the ML model (:code:`model_selection_cv` in the YAML specification).
This can be useful when a large number or range of hyperparameters is typically considered
(e.g., regularisation parameters in logistic regression).
Such additional cross-validation should be implemented inside the method :code:`_fit_by_cross_validation`.
The result should be that a single model (with optimal hyperparameters) is saved in the MLMethod object.
See :code:`SklearnMethod` for a detailed example.
**Note: this is advanced model implementation, which is usually not necessary to implement.**


Class documentation standards for ML methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ./class_documentation_standards.rst

.. collapse:: Click to view a full example of MLMethod class documentation.

       .. code::

        This SillyClassifier is a placeholder for a real ML method.
        It generates random predictions ignoring the input features.


        **Specification arguments:**

        - random_seed (int): The random seed for generating random predictions.


        **YAML specification:**

        .. indent with spaces
        .. code-block:: yaml

            my_silly_method:
                SillyClassifier:
                    random_seed: 100





