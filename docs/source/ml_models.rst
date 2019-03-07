########################
Machine learning models
########################

.. toctree::
   :maxdepth: 2

To perform machine learning (ML) on the immune receptor sequences, one should execute the following workflow:

1.  Load or simulate the immune receptor data in order to obtain a valid dataset,
2.  Encode the data, using one of :ref:`Encodings` and then
3.  Train one of the available models on the encoded data.

Machine learning models in ImmuneML provide classification, regression and clustering.

Classification
==============

In the case of classification, it is possible to perform binary classification, multi-class classification,
multi-label and multi-class multi-label classification.

For the case of immune status prediction, multi-label classification can be useful for predicting the status of multiple
diseases, while multi-class classification could be used to predict different stages of one specific disease. However,
in order to use this, the user needs only to supply labels for each of the diseases and an appropriate scenario will
be implemented internally.

The internal mechanism is the following:

*   In multi-class scenario, one classifier will perform the classification, which in turn might be implemented as one-vs-rest
    or inherently multi-class.
*   In multi-label scenario, one classifier will be fitted for each label performing the binary classification (i.e. the label
    is present or not).
*   In multi-label multi-class scenario, one classifier will be fitted for each label and perform multi-class
    classification for each label.

Running a machine learning algorithm
====================================

When performing machine learning as a part of an analysis, most often the **MLModelTrainer** class should be used.

.. code-block:: python

        trained_method = MLMethodTrainer.run({
            "method": method, # an untrained instance of a ML method, e.g. LogisticRegression() or SVM()
            "result_path": "../", # any path
            "dataset": dataset, # dataset used for training
            "labels": ["celiac", "cmv"], # list of names of labels which should be used for e.g. classification
            "number_of_splits": 10 # number of folds for k-fold cross-validation
        })

Internally, all machine learning model classes in ImmuneML inherit **MLModel** class and implement the following functions which
enable MLMethodTrainer class to train any of the methods uniformly:

.. code-block:: python

    def fit(self, X, y, label_names: list = None):
        pass

    def predict(self, X, label_names: list = None):
        pass

    def fit_by_cross_validation(self, X, y, number_of_splits: int = 5, parameter_grid: dict = None, label_names: list = None):
        pass

    def store(self, path):
        pass

    def load(self, path):
        pass

    def get_model(self, label_names: list = None):
        pass

    def check_if_exists(self, path):
        pass

In these functions, ``X`` is always a matrix (e.g. sparse matrix or numpy matrix) as returned by the encoders.
``y`` is a numpy matrix of labels where the first index determines the disease-specific label (e.g. celiac disease) and
the second index determines the value of the label for a given repertoire.

In case that the immune status is known for multiple diseases (e.g. celiac disease and CMV), it is possible to specify
a list of them for which a machine learning model should be built.

Parameter grid can be defined for each ML method as a dictionary of model hyper-parameters that should be explored and
chosen by cross-validation. For more information on possible values, see model descriptions on this page.

The full list of available machine learning models includes the following classes:

*   **LogisticRegression**
*   **SVM**
*   **RandomForestClassifier**

Algorithms
==========

========================
Logistic Regression
========================

Internally, **LogisticRegression** class is a wrapper around a *SGDClassifier* object with a logarithmic loss function.
*SGDClassifier* comes from ``scikit-learn`` library [1]_.

Possible parameters for fitting a logistic regression model include:

*   ``penalty``: can be "l1", "l2" or "elasticnet",
*   ``alpha``: regularization term, if not set, the default value is 0.0001,
*   ``fit_intercept``: can be True or False,
*   ``learning_rate``: can be "constant", "optimal", "invscaling" or "adaptive",
*   ``class_weight``: can be None, balanced or a dictionary with weights,
*   ``max_iter``: maximum number of passes over the training data.

For more information on these and other possible parameters, see sklearn's SGDClassifier_. An example of the parameter
grid for logistic regression is given below.

.. code-block:: python

    parameter_grid = {
        "max_iter": [10000, 12000],
        "penalty": ["l1", "l2"],
        "class_weight": ["balanced", None]
    }

The default values for the parameters for cross-validation are 10000 iterations, "l1" penalty and balanced classes.

Even though it is preferred to use **MLMethodTrainer** class to train ML models, it is also possible to do so manually.
The following code creates a logistic regression model in ImmuneML, fits the model to the encoded data and uses the model
to predict the disease status on another dataset.

.. code-block:: python

        from source.ml_methods.LogisticRegression import LogisticRegression

        train_dataset = ... # load encoded train dataset
        test_dataset  = ... # load encoded test dataset

        # create a logistic regression model
        logistic_regression_model = LogisticRegression()
        logistic_regression_model.fit_by_cross_validation(X=train_dataset.encoded_data["repertoires"],
                                                          y=train_dataset.encoded_data["labels"],
                                                          number_of_splits=10, # 10-fold cross-validation
                                                          parameter_grid={"max_iter": [10000, 12000],
                                                                          "penalty": ["l1", "l2"]},
                                                          label_names=["celiac", "cmv"]) # only for e.g. celiac disease and CMV
        # predict labels on the test dataset
        predictions = logistic_regression_model.predict(test_dataset.encoded_data["repertoires"])

In addition to examples and labels, if performing a fit by cross-validation, it is necessary to specify the number of
splits and the names of labels for which models need to be learnt form the data.

========================
Support Vector Machine
========================

**SVM** class is a wrapper class around sklearn's *SGDClassifier* class. Therefore, the same parameters apply as
for the **LogisticRegression** which is also based on *SGDClassifier*. For more information, see SGDClassifier_.
The difference between **SVM** and **LogisticRegression** classes is in the loss function they use:
**SVM** uses ``hinge`` loss function.

The default values in the parameter grid are the following:

.. code-block:: python

    {
        "max_iter": [150000],
        "penalty": ["l1"],
        "class_weight": ["balanced", None]
    }

An example of code creating a SVM model in ImmuneML, fitting the model to the encoded data and making predictions is
given below.

.. code-block:: python

        from source.ml_methods.SVM import SVM

        train_dataset = ... # load encoded train dataset
        test_dataset  = ... # load encoded test dataset

        # create a logistic regression model
        svm_model = SVM()
        svm_model.fit_by_cross_validation(X=train_dataset.encoded_data["repertoires"],
                                          y=train_dataset.encoded_data["labels"],
                                          number_of_splits=10, # 10-fold cross-validation
                                          parameter_grid={"max_iter": [10000, 12000], "penalty": ["l1", "l2"]},
                                          label_names=["celiac", "cmv"]) # only for e.g. celiac disease and CMV
        # predict labels on the test dataset
        predictions = svm_model.predict(test_dataset.encoded_data["repertoires"])

========================
Random Forest Classifier
========================

**RandomForestClassifier** class is a wrapper around sklearn's *RandomForestClassifier* [1]_. Parameters for this model are:

*   ``n_estimators``: number of decision trees in the forest (number of estimators for the ensemble method), with values
    most often between 10 and 100,
*   ``criterion``: criterion for measuring the quality of the split, possible values are "gini" and "entropy".

For more parameters and more detailed explanation, see RandomForestClassifier_ on sklearn's website.

Default values of these parameters in the parameter grid in the ImmuneML are defined only for the ``n_estimators``
and can be either 10, 50 or 100 estimators.

A code example using **RandomForestClassifier** is given below.

.. code-block:: python

        from source.ml_methods.RandomForestClassifier import RandomForestClassifier

        train_dataset = ... # load encoded train dataset
        test_dataset  = ... # load encoded test dataset

        # create a logistic regression model
        random_forest_model = RandomForestClassifier()
        random_forest_model.fit(X=train_dataset.encoded_data["repertoires"],
                                y=train_dataset.encoded_data["labels"],
                                label_names=["celiac", "cmv"]) # only for e.g. celiac disease and CMV
        # predict labels on the test dataset
        predictions = random_forest_model.predict(test_dataset.encoded_data["repertoires"])


.. [1] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

.. _SGDClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
.. _RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html