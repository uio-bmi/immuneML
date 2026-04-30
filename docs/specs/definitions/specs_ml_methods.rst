
**Classifiers**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


ML method classifiers are algorithms which can be trained to predict some label on immune
repertoires, receptors or sequences.

These methods can be trained using the :ref:`TrainMLModel` instruction, and previously trained
models can be applied to new data using the :ref:`MLApplication` instruction.

When choosing which ML method(s) are most suitable for your use-case, please consider the following table:

.. csv-table:: ML methods properties
   :file: ../../source/_static/files/ml_methods_properties.csv
   :header-rows: 1



BinaryFeatureClassifier
''''''''''''''''''''''''''''''''''''''''''''''''''''


A simple classifier that takes in encoded data containing features with only 1/0 or True/False values.

This classifier gives a positive prediction if any of the binary features for an example are 'true'.
Optionally, the classifier can select an optimal subset of these features. In this case, the given data is split
into a training and validation set, a minimal set of features is learned through greedy forward selection,
and the validation set is used to determine when to stop growing the set of features (earlystopping).
Earlystopping is reached when the optimization metric on the validation set no longer improves for a given number of features (patience).
The optimization metric is the same metric as the one used for optimization in the :py:obj:`~immuneML.workflows.instructions.TrainMLModelInstruction`.

Currently, this classifier can be used in combination with two encoders:

- The classifier can be used in combination with the :py:obj:`~immuneML.encodings.motif_encoding.MotifEncoder.MotifEncoder`,
  such that sequences containing any of the positive class-associated motifs are classified as positive.
  A reduced subset of binding-associated motifs can be learned (when keep_all is false).
  This results in a set of complementary motifs, minimizing the redundant predictions made by different motifs.

- Alternatively, this classifier can be combined with the :py:obj:`~immuneML.encodings.motif_encoding.SimilarToPositiveSequenceEncoder.SimilarToPositiveSequenceEncoder`
  such that any sequence that falls within a given hamming distance from any of the positive class sequences in the training set
  are classified as positive. Parameter keep_all should be set to true, since this encoder creates only 1 feature.


**Specification arguments:**

- training_percentage (float): What percentage of data to use for training (the rest will be used for validation); values between 0 and 1

- keep_all (bool): Whether to keep all the input features (true) or learn a reduced subset (false). By default, keep_all is false.

- random_seed (int): Random seed for splitting the data into training and validation sets when learning a minimal subset of features. This is only used when keep_all is false.

- max_features (int): The maximum number of features to allow in the reduced subset. When this number is reached, no more features are added even if the earlystopping criterion is not reached yet.
  This is only used when keep_all is false. By default, max_features is 100.

- patience (int): The patience for earlystopping. When earlystopping is reached, <patience> more features are added to the reduced set to test whether the optimization metric on the validation set improves again. By default, patience is 5.

- min_delta (float): The delta value used to test if there was improvement between the previous set of features and the new set of features (+1). By default, min_delta is 0, meaning the new set of features does not need to yield a higher optimization metric score on the validation set, but it needs to be at least equally high as the previous set.


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_motif_classifier:
                MotifClassifier:
                    training_percentage: 0.7
                    max_features: 100
                    patience: 5
                    min_delta: 0
                    keep_all: false



DeepRC
''''''''''''''''''''''''''''''''''''''''''''''''''''


This classifier uses the DeepRC method for repertoire classification. The DeepRC ML method should be used in combination
with the DeepRC encoder. Also consider using the :ref:`DeepRCMotifDiscovery` report for interpretability.

Notes:

- DeepRC uses PyTorch functionalities that depend on GPU. Therefore, DeepRC does not work on a CPU.

- This wrapper around DeepRC currently only supports binary classification.

Reference:
Michael Widrich, Bernhard Schäfl, Milena Pavlović, Geir Kjetil Sandve, Sepp Hochreiter, Victor Greiff, Günter Klambauer
‘DeepRC: Immune repertoire classification with attention-based deep massive multiple instance learning’.
bioRxiv preprint doi: `https://doi.org/10.1101/2020.04.12.038158 <https://doi.org/10.1101/2020.04.12.038158>`_


**Specification arguments:**

- validation_part (float):  the part of the data that will be used for validation, the rest will be used for training.

- add_positional_information (bool): whether positional information should be included in the input features.

- kernel_size (int): the size of the 1D-CNN kernels.

- n_kernels (int): the number of 1D-CNN kernels in each layer.

- n_additional_convs (int): Number of additional 1D-CNN layers after first layer

- n_attention_network_layers (int): Number of attention layers to compute keys

- n_attention_network_units (int): Number of units in each attention layer

- n_output_network_units (int): Number of units in the output layer

- consider_seq_counts (bool): whether the input data should be scaled by the receptor sequence counts.

- sequence_reduction_fraction (float): Fraction of number of sequences to which to reduce the number of sequences per bag based on attention weights. Has to be in range [0,1].

- reduction_mb_size (int): Reduction of sequences per bag is performed using minibatches of reduction_mb_size` sequences to compute the attention weights.

- n_updates (int): Number of updates to train for

- n_torch_threads (int):  Number of parallel threads to allow PyTorch

- learning_rate (float): Learning rate for adam optimizer

- l1_weight_decay (float): l1 weight decay factor. l1 weight penalty will be added to loss, scaled by `l1_weight_decay`

- l2_weight_decay (float): l2 weight decay factor. l2 weight penalty will be added to loss, scaled by `l2_weight_decay`

- sequence_counts_scaling_fn: it can either be `log` (logarithmic scaling of sequence counts) or None

- sequence_counts_scaling_fn: it can either be `log` (logarithmic scaling of sequence counts) or None

- evaluate_at (int): Evaluate model on training and validation set every `evaluate_at` updates. This will also check for a new best model for early stopping.

- sample_n_sequences (int): Optional random sub-sampling of `sample_n_sequences` sequences per repertoire. Number of sequences per repertoire might be smaller than `sample_n_sequences` if repertoire is smaller or random indices have been drawn multiple times. If None, all sequences will be loaded for each repertoire.

- training_batch_size (int): Number of repertoires per minibatch during training.

- n_workers (int): Number of background processes to use for converting dataset to hdf5 container and training set data loader.

- pytorch_device_name (str): The name of the pytorch device to use. This name will be passed to  torch.device(self.pytorch_device_name). The default value is cuda:0


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_deeprc_method:
                DeepRC:
                    validation_part: 0.2
                    add_positional_information: True
                    kernel_size: 9



GradientBoosting
''''''''''''''''''''''''''''''''''''''''''''''''''''


Gradient Boosting classifier which wraps scikit-learn's GradientBoostingClassifier.
Input arguments for the method are the same as supported by scikit-learn (see `GradientBoostingClassifier scikit-learn documentation
<https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_ for details).



Scikit-learn models can be trained in two modes: 

1. Creating a model using a given set of hyperparameters, and relying on the selection and assessment loop in the
TrainMLModel instruction to select the optimal model. 

2. Passing a range of different hyperparameters to GradientBoosting, and using a third layer of nested cross-validation 
to find the optimal hyperparameters through grid search. In this case, only the GradientBoosting model with the optimal 
hyperparameter settings is further used in the inner selection loop of the TrainMLModel instruction. 

By default, mode 1 is used. In order to use mode 2, model_selection_cv and model_selection_n_folds must be set. 


**Specification arguments:**

- GradientBoosting (dict): Under this key, hyperparameters can be specified that will be passed to the scikit-learn class.
  Any scikit-learn hyperparameters can be specified here. In mode 1, a single value must be specified for each of the scikit-learn
  hyperparameters. In mode 2, it is possible to specify a range of different hyperparameters values in a list. It is also allowed
  to mix lists and single values in mode 2, in which case the grid search will only be done for the lists, while the
  single-value hyperparameters will be fixed. 
  In addition to the scikit-learn hyperparameters, parameter show_warnings (True/False) can be specified here. This determines
  whether scikit-learn warnings, such as convergence warnings, should be printed. By default show_warnings is True.
    
- model_selection_cv (bool): If any of the hyperparameters under GradientBoosting is a list and model_selection_cv is True, 
  a grid search will be done over the given hyperparameters, using the number of folds specified in model_selection_n_folds.
  By default, model_selection_cv is False. 
    
- model_selection_n_folds (int): The number of folds that should be used for the cross validation grid search if model_selection_cv is True.
    


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_gradient_boosting:
                GradientBoosting:
                    # arguments as defined by scikit-learn
                    n_estimators: 100
                    learning_rate: 0.1
                    max_depth: 3
                    random_state: 42

    

KNN
''''''''''''''''''''''''''''''''''''''''''''''''''''


This is a wrapper of scikit-learn’s KNeighborsClassifier class.
This ML method creates a distance matrix using the given encoded data. If the encoded data is already a distance
matrix (for example, when using the :ref:`Distance` or :ref:`CompAIRRDistance` encoders), please use :ref:`PrecomputedKNN` instead.

Please see the `KNeighborsClassifier scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_
of KNeighborsClassifier for the parameters.



Scikit-learn models can be trained in two modes: 

1. Creating a model using a given set of hyperparameters, and relying on the selection and assessment loop in the
TrainMLModel instruction to select the optimal model. 

2. Passing a range of different hyperparameters to KNN, and using a third layer of nested cross-validation 
to find the optimal hyperparameters through grid search. In this case, only the KNN model with the optimal 
hyperparameter settings is further used in the inner selection loop of the TrainMLModel instruction. 

By default, mode 1 is used. In order to use mode 2, model_selection_cv and model_selection_n_folds must be set. 


**Specification arguments:**

- KNN (dict): Under this key, hyperparameters can be specified that will be passed to the scikit-learn class.
  Any scikit-learn hyperparameters can be specified here. In mode 1, a single value must be specified for each of the scikit-learn
  hyperparameters. In mode 2, it is possible to specify a range of different hyperparameters values in a list. It is also allowed
  to mix lists and single values in mode 2, in which case the grid search will only be done for the lists, while the
  single-value hyperparameters will be fixed. 
  In addition to the scikit-learn hyperparameters, parameter show_warnings (True/False) can be specified here. This determines
  whether scikit-learn warnings, such as convergence warnings, should be printed. By default show_warnings is True.
    
- model_selection_cv (bool): If any of the hyperparameters under KNN is a list and model_selection_cv is True, 
  a grid search will be done over the given hyperparameters, using the number of folds specified in model_selection_n_folds.
  By default, model_selection_cv is False. 
    
- model_selection_n_folds (int): The number of folds that should be used for the cross validation grid search if model_selection_cv is True.
    


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_knn_method:
                KNN:
                    # sklearn parameters (same names as in original sklearn class)
                    weights: uniform # always use this setting for weights
                    n_neighbors: [5, 10, 15] # find the optimal number of neighbors
                    # Additional parameter that determines whether to print convergence warnings
                    show_warnings: True
                # if any of the parameters under KNN is a list and model_selection_cv is True,
                # a grid search will be done over the given parameters, using the number of folds specified in model_selection_n_folds,
                # and the optimal model will be selected
                model_selection_cv: True
                model_selection_n_folds: 5
            # alternative way to define ML method with default values:
            my_default_knn: KNN



KerasSequenceCNN
''''''''''''''''''''''''''''''''''''''''''''''''''''


A CNN-based classifier for sequence datasets. Should be used in combination with :py:obj:`source.encodings.onehot.OneHotEncoder.OneHotEncoder`.
This classifier integrates the CNN proposed by Mason et al., the original code can be found at: https://github.com/dahjan/DMS_opt/blob/master/scripts/CNN.py

Note: make sure keras and tensorflow dependencies are installed (see installation instructions).

Reference:
Derek M. Mason, Simon Friedensohn, Cédric R. Weber, Christian Jordi, Bastian Wagner, Simon M. Men1, Roy A. Ehling,
Lucia Bonati, Jan Dahinden, Pablo Gainza, Bruno E. Correia and Sai T. Reddy
‘Optimization of therapeutic antibodies by predicting antigen specificity from antibody sequence via deep learning’.
Nat Biomed Eng 5, 600–612 (2021). https://doi.org/10.1038/s41551-021-00699-9

**Specification arguments:**

- units_per_layer (list): A nested list specifying the layers of the CNN. The first element in each nested list defines the layer type, other elements define the layer parameters.
  Valid layer types are: CONV (keras.layers.Conv1D), DROP (keras.layers.Dropout), POOL (keras.layers.MaxPool1D), FLAT (keras.layers.Flatten), DENSE (keras.layers.Dense).
  The parameters per layer type are as follows:

    - [CONV, <filters>, <kernel_size>, <strides>]

    - [DROP, <rate>]

    - [POOL, <pool_size>, <strides>]

    - [FLAT]

    - [DENSE, <units>]

- activation (str): The Activation function to use in the convolutional or dense layers. Activation functions can be chosen from keras.activations. For example, rely or softmax. By default, relu is used.

- training_percentage (float): The fraction of sequences that will be randomly assigned to form the training set (the rest will be the validation set). Should be a value between 0 and 1. By default, training_percentage is 0.7.


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_cnn:
                KerasSequenceCNN:
                    training_percentage: 0.7
                    units_per_layer: [[CONV, 400, 3, 1], [DROP, 0.5], [POOL, 2, 1], [FLAT], [DENSE, 50]]
                    activation: relu





LogRegressionCustomPenalty
''''''''''''''''''''''''''''''''''''''''''''''''''''


Logistic Regression with custom penalty factors for specific features.

**Specification arguments**:

- non_penalized_features (list): List of feature names that should not be penalized.

- non_penalized_encodings (list): List of encoding names (class names) whose features should not be penalized. This
  parameter can be used only in combination with CompositeEncoder. None fo the features from the specified encodings
  will be penalized. If both non_penalized_features and non_penalized_encodings are provided, the union of the two
  will be used.

Other supported arguments are inherited from LogitNet of python-glmnet package and will be directly passed to it.
n_jobs will be overwritten to use the number of CPUs specified for the instruction (e.g. in TrainMLModel).

**YAML specification:**

.. code-block:: yaml

    ml_methods:
        custom_log_reg:
            LogRegressionCustomPenalty:
                alpha: 1
                n_lambda: 100
                non_penalized_features: []
                non_penalized_encodings: ['Metadata']
                random_state: 42



LogisticRegression
''''''''''''''''''''''''''''''''''''''''''''''''''''


This is a wrapper of scikit-learn’s LogisticRegression class. Please see the
`LogisticRegression scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
of LogisticRegression for the parameters.

Note: if you are interested in plotting the coefficients of the logistic regression model,
consider running the :ref:`Coefficients` report.



Scikit-learn models can be trained in two modes: 

1. Creating a model using a given set of hyperparameters, and relying on the selection and assessment loop in the
TrainMLModel instruction to select the optimal model. 

2. Passing a range of different hyperparameters to LogisticRegression, and using a third layer of nested cross-validation 
to find the optimal hyperparameters through grid search. In this case, only the LogisticRegression model with the optimal 
hyperparameter settings is further used in the inner selection loop of the TrainMLModel instruction. 

By default, mode 1 is used. In order to use mode 2, model_selection_cv and model_selection_n_folds must be set. 


**Specification arguments:**

- LogisticRegression (dict): Under this key, hyperparameters can be specified that will be passed to the scikit-learn class.
  Any scikit-learn hyperparameters can be specified here. In mode 1, a single value must be specified for each of the scikit-learn
  hyperparameters. In mode 2, it is possible to specify a range of different hyperparameters values in a list. It is also allowed
  to mix lists and single values in mode 2, in which case the grid search will only be done for the lists, while the
  single-value hyperparameters will be fixed. 
  In addition to the scikit-learn hyperparameters, parameter show_warnings (True/False) can be specified here. This determines
  whether scikit-learn warnings, such as convergence warnings, should be printed. By default show_warnings is True.
    
- model_selection_cv (bool): If any of the hyperparameters under LogisticRegression is a list and model_selection_cv is True, 
  a grid search will be done over the given hyperparameters, using the number of folds specified in model_selection_n_folds.
  By default, model_selection_cv is False. 
    
- model_selection_n_folds (int): The number of folds that should be used for the cross validation grid search if model_selection_cv is True.
    



**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_logistic_regression: # user-defined method name
                LogisticRegression: # name of the ML method
                    # sklearn parameters (same names as in original sklearn class)
                    penalty: l1 # always use penalty l1
                    C: [0.01, 0.1, 1, 10, 100] # find the optimal value for C
                    # Additional parameter that determines whether to print convergence warnings
                    show_warnings: True
                # if any of the parameters under LogisticRegression is a list and model_selection_cv is True,
                # a grid search will be done over the given parameters, using the number of folds specified in model_selection_n_folds,
                # and the optimal model will be selected
                model_selection_cv: True
                model_selection_n_folds: 5
            # alternative way to define ML method with default values:
            my_default_logistic_regression: LogisticRegression



PrecomputedKNN
''''''''''''''''''''''''''''''''''''''''''''''''''''


This is a wrapper of scikit-learn’s KNeighborsClassifier class.
This ML method takes a pre-computed distance matrix, as created by the :ref:`Distance` or :ref:`CompAIRRDistance` encoders.
If you would like to use a different encoding in combination with KNN, please use :ref:`KNN` instead.

Please see the `KNN scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_
of KNeighborsClassifier for the parameters.



Scikit-learn models can be trained in two modes: 

1. Creating a model using a given set of hyperparameters, and relying on the selection and assessment loop in the
TrainMLModel instruction to select the optimal model. 

2. Passing a range of different hyperparameters to KNN, and using a third layer of nested cross-validation 
to find the optimal hyperparameters through grid search. In this case, only the KNN model with the optimal 
hyperparameter settings is further used in the inner selection loop of the TrainMLModel instruction. 

By default, mode 1 is used. In order to use mode 2, model_selection_cv and model_selection_n_folds must be set. 


**Specification arguments:**

- KNN (dict): Under this key, hyperparameters can be specified that will be passed to the scikit-learn class.
  Any scikit-learn hyperparameters can be specified here. In mode 1, a single value must be specified for each of the scikit-learn
  hyperparameters. In mode 2, it is possible to specify a range of different hyperparameters values in a list. It is also allowed
  to mix lists and single values in mode 2, in which case the grid search will only be done for the lists, while the
  single-value hyperparameters will be fixed. 
  In addition to the scikit-learn hyperparameters, parameter show_warnings (True/False) can be specified here. This determines
  whether scikit-learn warnings, such as convergence warnings, should be printed. By default show_warnings is True.
    
- model_selection_cv (bool): If any of the hyperparameters under KNN is a list and model_selection_cv is True, 
  a grid search will be done over the given hyperparameters, using the number of folds specified in model_selection_n_folds.
  By default, model_selection_cv is False. 
    
- model_selection_n_folds (int): The number of folds that should be used for the cross validation grid search if model_selection_cv is True.
    



**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_knn_method:
                PrecomputedKNN:
                    # sklearn parameters (same names as in original sklearn class)
                    weights: uniform # always use this setting for weights
                    n_neighbors: [5, 10, 15] # find the optimal number of neighbors
                    # Additional parameter that determines whether to print convergence warnings
                    show_warnings: True
                # if any of the parameters under KNN is a list and model_selection_cv is True,
                # a grid search will be done over the given parameters, using the number of folds specified in model_selection_n_folds,
                # and the optimal model will be selected
                model_selection_cv: True
                model_selection_n_folds: 5
            # alternative way to define ML method with default values:
            my_default_knn: PrecomputedKNN



ProbabilisticBinaryClassifier
''''''''''''''''''''''''''''''''''''''''''''''''''''


ProbabilisticBinaryClassifier predicts the class assignment in binary classification case based on encoding examples by number of
successful trials and total number of trials. It models this ratio by one beta distribution per class and predicts the class of the new
examples using log-posterior odds ratio with threshold at 0.

ProbabilisticBinaryClassifier is based on the paper (details on the classification can be found in the Online Methods section):
Emerson, Ryan O., William S. DeWitt, Marissa Vignali, Jenna Gravley, Joyce K. Hu, Edward J. Osborne, Cindy Desmarais, et al.
‘Immunosequencing Identifies Signatures of Cytomegalovirus Exposure History and HLA-Mediated Effects on the T Cell Repertoire’.
Nature Genetics 49, no. 5 (May 2017): 659–65. `doi.org/10.1038/ng.3822 <https://doi.org/10.1038/ng.3822>`_.

**Specification arguments:**

- max_iterations (int): maximum number of iterations while optimizing the parameters of the beta distribution (same for both classes)

- update_rate (float): how much the computed gradient should influence the updated value of the parameters of the beta distribution

- likelihood_threshold (float): at which threshold to stop the optimization (default -1e-10)

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_probabilistic_classifier: # user-defined name of the ML method
                ProbabilisticBinaryClassifier: # method name
                    max_iterations: 1000
                    update_rate: 0.01



RandomForestClassifier
''''''''''''''''''''''''''''''''''''''''''''''''''''


This is a wrapper of scikit-learn’s RandomForestClassifier class. Please see the
`RandomForestClassifier scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_
of RandomForestClassifier for the parameters.

Note: if you are interested in plotting the coefficients of the random forest classifier model,
consider running the :ref:`Coefficients` report.



Scikit-learn models can be trained in two modes: 

1. Creating a model using a given set of hyperparameters, and relying on the selection and assessment loop in the
TrainMLModel instruction to select the optimal model. 

2. Passing a range of different hyperparameters to RandomForestClassifier, and using a third layer of nested cross-validation 
to find the optimal hyperparameters through grid search. In this case, only the RandomForestClassifier model with the optimal 
hyperparameter settings is further used in the inner selection loop of the TrainMLModel instruction. 

By default, mode 1 is used. In order to use mode 2, model_selection_cv and model_selection_n_folds must be set. 


**Specification arguments:**

- RandomForestClassifier (dict): Under this key, hyperparameters can be specified that will be passed to the scikit-learn class.
  Any scikit-learn hyperparameters can be specified here. In mode 1, a single value must be specified for each of the scikit-learn
  hyperparameters. In mode 2, it is possible to specify a range of different hyperparameters values in a list. It is also allowed
  to mix lists and single values in mode 2, in which case the grid search will only be done for the lists, while the
  single-value hyperparameters will be fixed. 
  In addition to the scikit-learn hyperparameters, parameter show_warnings (True/False) can be specified here. This determines
  whether scikit-learn warnings, such as convergence warnings, should be printed. By default show_warnings is True.
    
- model_selection_cv (bool): If any of the hyperparameters under RandomForestClassifier is a list and model_selection_cv is True, 
  a grid search will be done over the given hyperparameters, using the number of folds specified in model_selection_n_folds.
  By default, model_selection_cv is False. 
    
- model_selection_n_folds (int): The number of folds that should be used for the cross validation grid search if model_selection_cv is True.
    



**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_random_forest_classifier: # user-defined method name
                RandomForestClassifier: # name of the ML method
                    # sklearn parameters (same names as in original sklearn class)
                    random_state: 100 # always use this value for random state
                    n_estimators: [10, 50, 100] # find the optimal number of trees in the forest
                    # Additional parameter that determines whether to print convergence warnings
                    show_warnings: True
                # if any of the parameters under RandomForestClassifier is a list and model_selection_cv is True,
                # a grid search will be done over the given parameters, using the number of folds specified in model_selection_n_folds,
                # and the optimal model will be selected
                model_selection_cv: True
                model_selection_n_folds: 5
            # alternative way to define ML method with default values:
            my_default_random_forest: RandomForestClassifier



ReceptorCNN
''''''''''''''''''''''''''''''''''''''''''''''''''''


A CNN which separately detects motifs using CNN kernels in each chain of paired receptor data, combines the kernel activations into a unique
representation of the receptor and uses this representation to predict the antigen binding.

.. figure:: ../_static/images/receptor_cnn_immuneML.png
    :width: 70%

    The architecture of the CNN for paired-chain receptor data

Requires one-hot encoded data as input (as produced by :ref:`OneHot` encoder), where use_positional_info must be set to True.

Notes:

- ReceptorCNN can only be used with ReceptorDatasets, it does not work with SequenceDatasets

- ReceptorCNN can only be used for binary classification, not multi-class classification.


**Specification arguments:**

- kernel_count (count): number of kernels that will look for motifs for one chain

- kernel_size (list): sizes of the kernels = how many amino acids to consider at the same time in the chain sequence, can be a tuple of values; e.g. for value [3, 4] of kernel_size, kernel_count*len(kernel_size) kernels will be created, with kernel_count kernels of size 3 and kernel_count kernels of size 4 per chain

- positional_channels (int): how many positional channels where included in one-hot encoding of the receptor sequences (:ref:`OneHot` encoder adds 3 positional channels positional information is enabled)

- sequence_type (SequenceType): type of the sequence

- device: which device to use for the model (cpu or gpu) - for more details see PyTorch documentation on device parameter

- number_of_threads (int): how many threads to use

- random_seed (int): number used as a seed for random initialization

- learning_rate (float): learning rate scaling the step size for optimization algorithm

- iteration_count (int): for how many iterations to train the model

- l1_weight_decay (float): weight decay l1 value for the CNN; encourages sparser representations

- l2_weight_decay (float): weight decay l2 value for the CNN; shrinks weight coefficients towards zero

- batch_size (int): how many receptors to process at once

- training_percentage (float): what percentage of data to use for training (the rest will be used for validation); values between 0 and 1

- evaluate_at (int): when to evaluate the model, e.g. every 100 iterations

- background_probabilities: used for rescaling the kernel values to produce information gain matrix; represents the background probability of each amino acid (without positional information); if not specified, uniform background is assumed

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_receptor_cnn:
                ReceptorCNN:
                    kernel_count: 5
                    kernel_size: [3]
                    positional_channels: 3
                    sequence_type: amino_acid
                    device: cpu
                    number_of_threads: 16
                    random_seed: 100
                    learning_rate: 0.01
                    iteration_count: 10000
                    l1_weight_decay: 0
                    l2_weight_decay: 0
                    batch_size: 5000



SVC
''''''''''''''''''''''''''''''''''''''''''''''''''''


This is a wrapper of scikit-learn’s LinearSVC class. Please see the
`LinearSVC scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html>`_
of SVC for the parameters.

Note: if you are interested in plotting the coefficients of the SVC model,
consider running the :ref:`Coefficients` report.



Scikit-learn models can be trained in two modes: 

1. Creating a model using a given set of hyperparameters, and relying on the selection and assessment loop in the
TrainMLModel instruction to select the optimal model. 

2. Passing a range of different hyperparameters to SVC, and using a third layer of nested cross-validation 
to find the optimal hyperparameters through grid search. In this case, only the SVC model with the optimal 
hyperparameter settings is further used in the inner selection loop of the TrainMLModel instruction. 

By default, mode 1 is used. In order to use mode 2, model_selection_cv and model_selection_n_folds must be set. 


**Specification arguments:**

- SVC (dict): Under this key, hyperparameters can be specified that will be passed to the scikit-learn class.
  Any scikit-learn hyperparameters can be specified here. In mode 1, a single value must be specified for each of the scikit-learn
  hyperparameters. In mode 2, it is possible to specify a range of different hyperparameters values in a list. It is also allowed
  to mix lists and single values in mode 2, in which case the grid search will only be done for the lists, while the
  single-value hyperparameters will be fixed. 
  In addition to the scikit-learn hyperparameters, parameter show_warnings (True/False) can be specified here. This determines
  whether scikit-learn warnings, such as convergence warnings, should be printed. By default show_warnings is True.
    
- model_selection_cv (bool): If any of the hyperparameters under SVC is a list and model_selection_cv is True, 
  a grid search will be done over the given hyperparameters, using the number of folds specified in model_selection_n_folds.
  By default, model_selection_cv is False. 
    
- model_selection_n_folds (int): The number of folds that should be used for the cross validation grid search if model_selection_cv is True.
    



**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_svc: # user-defined method name
                SVC: # name of the ML method
                    # sklearn parameters (same names as in original sklearn class)
                    C: [0.01, 0.1, 1, 10, 100] # find the optimal value for C
                    # Additional parameter that determines whether to print convergence warnings
                    show_warnings: True
                # if any of the parameters under SVC is a list and model_selection_cv is True,
                # a grid search will be done over the given parameters, using the number of folds specified in model_selection_n_folds,
                # and the optimal model will be selected
                model_selection_cv: True
                model_selection_n_folds: 5
            # alternative way to define ML method with default values:
            my_default_svc: SVC



SVM
''''''''''''''''''''''''''''''''''''''''''''''''''''


This is a wrapper of scikit-learn’s SVC class. Please see the
`SVC scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_
of SVC for the parameters.

Note: if you are interested in plotting the coefficients of the SVM model,
consider running the :ref:`Coefficients` report.



Scikit-learn models can be trained in two modes: 

1. Creating a model using a given set of hyperparameters, and relying on the selection and assessment loop in the
TrainMLModel instruction to select the optimal model. 

2. Passing a range of different hyperparameters to SVM, and using a third layer of nested cross-validation 
to find the optimal hyperparameters through grid search. In this case, only the SVM model with the optimal 
hyperparameter settings is further used in the inner selection loop of the TrainMLModel instruction. 

By default, mode 1 is used. In order to use mode 2, model_selection_cv and model_selection_n_folds must be set. 


**Specification arguments:**

- SVM (dict): Under this key, hyperparameters can be specified that will be passed to the scikit-learn class.
  Any scikit-learn hyperparameters can be specified here. In mode 1, a single value must be specified for each of the scikit-learn
  hyperparameters. In mode 2, it is possible to specify a range of different hyperparameters values in a list. It is also allowed
  to mix lists and single values in mode 2, in which case the grid search will only be done for the lists, while the
  single-value hyperparameters will be fixed. 
  In addition to the scikit-learn hyperparameters, parameter show_warnings (True/False) can be specified here. This determines
  whether scikit-learn warnings, such as convergence warnings, should be printed. By default show_warnings is True.
    
- model_selection_cv (bool): If any of the hyperparameters under SVM is a list and model_selection_cv is True, 
  a grid search will be done over the given hyperparameters, using the number of folds specified in model_selection_n_folds.
  By default, model_selection_cv is False. 
    
- model_selection_n_folds (int): The number of folds that should be used for the cross validation grid search if model_selection_cv is True.
    



**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_svm: # user-defined method name
                SVM: # name of the ML method
                    # sklearn parameters (same names as in original sklearn class)
                    C: [0.01, 0.1, 1, 10, 100] # find the optimal value for C
                    kernel: linear
                    # Additional parameter that determines whether to print convergence warnings
                    show_warnings: True
                # if any of the parameters under SVM is a list and model_selection_cv is True,
                # a grid search will be done over the given parameters, using the number of folds specified in model_selection_n_folds,
                # and the optimal model will be selected
                model_selection_cv: True
                model_selection_n_folds: 5
            # alternative way to define ML method with default values:
            my_default_svm: SVM



TCRdistClassifier
''''''''''''''''''''''''''''''''''''''''''''''''''''


Implementation of a nearest neighbors classifier based on TCR distances as presented in
Dash P, Fiore-Gartland AJ, Hertz T, et al. Quantifiable predictive features define epitope-specific T cell receptor repertoires.
Nature. 2017; 547(7661):89-93. `doi:10.1038/nature22383 <https://www.nature.com/articles/nature22383>`_.

This method is implemented using scikit-learn's KNeighborsClassifier with k determined at runtime from the training dataset size and weights
linearly scaled to decrease with the distance of examples.

**Specification arguments:**

- percentage (float): percentage of nearest neighbors to consider when determining receptor specificity based on known receptors (between 0 and 1)

- show_warnings (bool): whether to show warnings generated by scikit-learn, by default this is True.

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_tcr_method:
                TCRdistClassifier:
                    percentage: 0.1
                    show_warnings: True



**Clustering methods**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



Clustering methods are algorithms which can be used to cluster repertoires, receptors or
sequences without using external label information (such as disease or antigen binding state)

These methods can be used in the :ref:`Clustering` instruction.



AgglomerativeClustering
''''''''''''''''''''''''''''''''''''''''''''''''''''


Agglomerative clustering method which wraps scikit-learn's clustering of the same name.
Input arguments for the method are the same as supported by scikit-learn (see `AgglomerativeClustering scikit-learn documentation
<https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>`_ for details).

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_agglomerative_clustering:
                AgglomerativeClustering:
                    # arguments as defined by scikit-learn
                    n_clusters: 3
                    linkage: 'ward'
    

DBSCAN
''''''''''''''''''''''''''''''''''''''''''''''''''''


DBSCAN method which wraps scikit-learn's clustering of the same name.
Input arguments for the method are the same as supported by scikit-learn (see `DBSCAN scikit-learn documentation
<https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_ for details).

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_dbscan:
                DBSCAN:
                    # arguments as defined by scikit-learn
                    eps: 0.5
                    min_samples: 5
    

HDBSCAN
''''''''''''''''''''''''''''''''''''''''''''''''''''


HDBSCAN method which wraps scikit-learn's clustering of the same name.
Input arguments for the method are the same as supported by scikit-learn (see `DBSCAN scikit-learn documentation
<https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html>`_ for details).

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_hdbscan:
                HDBSCAN:
                    # arguments as defined by scikit-learn
                    min_cluster_size: 5

    

KMeans
''''''''''''''''''''''''''''''''''''''''''''''''''''


k-means clustering method which wraps scikit-learn's KMeans. Input arguments for the method are the
same as supported by scikit-learn (see `KMeans scikit-learn documentation
<https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_ for details).

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_kmeans:
                KMeans:
                    # arguments as defined by scikit-learn
                    n_clusters: 2


**Generative models**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



Generative models are algorithms which can be trained to learn patterns in existing datasets,
and then be used to generate new synthetic datasets.

These methods can be used in the :ref:`TrainGenModel` instruction, and previously trained
models can be used to generate data using the :ref:`ApplyGenModel` instruction.


ExperimentalImport
''''''''''''''''''''''''''''''''''''''''''''''''''''


Allows to import existing experimental data and do annotations and simulations on top of them.
This model should be used only for LIgO simulation and not with TrainGenModel instruction.

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            generative_model:
                type: ExperimentalImport
                import_format: AIRR
                tmp_import_path: ./tmp/
                import_params:
                    path: path/to/files/
                    region_type: IMGT_CDR3 # what part of the sequence to import
                    column_mapping: # column mapping AIRR: immuneML
                        junction: sequence
                        junction_aa: sequence_aa
                        locus: chain


OLGA
''''''''''''''''''''''''''''''''''''''''''''''''''''


This is a wrapper for the OLGA package as described by Sethna et al. 2019 (OLGA package on PyPI or GitHub:
https://github.com/statbiophys/OLGA ).
This model should be used only for LIgO simulation and is not yet supported for use with TrainGenModel instruction.


Reference:

Zachary Sethna, Yuval Elhanati, Curtis G Callan, Jr, Aleksandra M Walczak, Thierry Mora, OLGA: fast computation of
generation probabilities of B- and T-cell receptor amino acid sequences and motifs, Bioinformatics, Volume 35,
Issue 17, 1 September 2019, Pages 2974–2981, https://doi.org/10.1093/bioinformatics/btz035

Note:

- OLGA generates sequences that correspond to IMGT junction and are used for matching as such. See the
  https://github.com/statbiophys/OLGA for more details.

- Gene names are as provided in OLGA (either in default models or in the user-specified model files). For
  simulation, one should use gene names in the same format.

.. note::

    While this is a generative model, in the current version of immuneML it cannot be used in combination with TrainGenModel or
    ApplyGenModel instruction. If you want to use OLGA for sequence simulation, see :ref:`Dataset simulation with LIgO`.
`
**Specification arguments:**

- model_path (str): if not default model, this parameter should point to a folder where the four OLGA/IGOR format
  files are stored (could also be inferred from some experimental data)

- default_model_name (str): if not using custom models, one of the OLGA default models could be specified here;
  the value should be the same as it would be passed to command line in OLGA: e.g., humanTRB, human IGH

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            generative_model:
                type: OLGA
                model_path: None
                default_model_name: humanTRB



PWM
''''''''''''''''''''''''''''''''''''''''''''''''''''



This is a baseline implementation of a positional weight matrix. It is estimated from a set of sequences for each
of the different lengths that appear in the dataset.


**Specification arguments:**

- locus (str): which chain is generated (for now, it is only assigned to the generated sequences)

- sequence_type (str): amino_acid or nucleotide

- region_type (str): which region type to use (e.g., IMGT_CDR3), this is only assigned to the generated sequences


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_pwm:
                PWM:
                    locus: beta
                    sequence_type: amino_acid
                    region_type: IMGT_CDR3



ProGen
''''''''''''''''''''''''''''''''''''''''''''''''''''


ProGen is a transformer-based language model for protein sequences. This class allows fine-tuning of a pre-trained
ProGen model on immune receptor sequences and generating new sequences. It is based on the ProGen2 implementation
available at https://github.com/salesforce/progen. It uses the sequences as given in "junction_aa" field in the
input dataset.

References:

Nijkamp, E., Ruffolo, J. A., Weinstein, E. N., Naik, N., & Madani, A. (2023).
Exploring the boundaries of protein language models. Cell Systems, 14(11), 968–978.e3.
https://doi.org/10.1016/j.cels.2023.10.002

**Specification arguments:**

- locus (str): which locus the sequence come from, e.g., TRB

- tokenizer_path (Path): path to the ProGen tokenizer file (tokenizer.json)

- trained_model_path (Path): path to the pre-trained ProGen model directory

- num_frozen_layers (int): number of transformer layers to freeze during fine-tuning

- num_epochs (int): number of epochs for fine-tuning

- learning_rate (float): learning rate for fine-tuning

- device (str): device to use for training and inference ("cpu" or "cuda")

- fp16 (bool): whether to use mixed precision training

- prefix_text (str): text to prepend to each sequence during fine-tuning

- suffix_text (str): text to append to each sequence during fine-tuning

- max_new_tokens (int): maximum number of new tokens to generate

- temperature (float): sampling temperature for sequence generation

- top_p (float): nucleus sampling parameter for sequence generation

- prompt (str): prompt text to start the generation

- num_gen_batches (int): number of batches to split generation into

- per_device_train_batch_size (int): batch size per device during fine-tuning

- remove_affixes (bool): whether to remove prefix and suffix from generated sequences

- seed (int): random seed for reproducibility


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            progen_model:
                ProGen:
                    locus: 'beta'
                    tokenizer_path: '/path/to/tokenizer.json'
                    trained_model_path: '/path/to/pretrained/progen/model'
                    num_frozen_layers: 27
                    num_epochs: 3
                    learning_rate: 0.00004
                    device: 'cuda'
                    fp16: False
                    prefix_text: '<|bos|>1'
                    suffix_text: '2<|eos|>'
                    max_new_tokens: 1024
                    temperature: 1.0
                    top_p: 0.9
                    prompt: '1'
                    num_gen_batches: 1
                    per_device_train_batch_size: 2
                    remove_affixes: True
                    name: 'progen_finetuned_model'
                    region_type: 'IMGT_JUNCTION'
                    seed: 42



SimpleLSTM
''''''''''''''''''''''''''''''''''''''''''''''''''''


This is a simple generative model for receptor sequences based on LSTM.

Similar models have been proposed in:

Akbar, R. et al. (2022). In silico proof of principle of machine learning-based antibody design at unconstrained scale. mAbs, 14(1), 2031482. https://doi.org/10.1080/19420862.2022.2031482

Saka, K. et al. (2021). Antibody design using LSTM based deep generative model from phage display library for affinity maturation. Scientific Reports, 11(1), Article 1. https://doi.org/10.1038/s41598-021-85274-7


**Specification arguments:**

- sequence_type (str): whether the model should work on amino_acid or nucleotide level

- hidden_size (int): how many LSTM cells should exist per layer

- num_layers (int): how many hidden LSTM layers should there be

- num_epochs (int): for how many epochs to train the model

- learning_rate (float): what learning rate to use for optimization

- batch_size (int): how many examples (sequences) to use for training for one batch

- embed_size (int): the dimension of the sequence embedding

- temperature (float): a higher temperature leads to faster yet more unstable learning

- prime_str (str): the initial sequence to start generating from

- seed (int): random seed for the model or None

- iter_to_report (int): number of epochs between training progress reports


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_simple_lstm:
                sequence_type: amino_acid
                hidden_size: 50
                num_layers: 1
                num_epochs: 5000
                learning_rate: 0.001
                batch_size: 100
                embed_size: 100




SimpleVAE
''''''''''''''''''''''''''''''''''''''''''''''''''''


SimpleVAE is a generative model on sequence level that relies on variational autoencoder. This type of model was
proposed by Davidsen et al. 2019, and this implementation is inspired by their original implementation available
at https://github.com/matsengrp/vampire. It uses the sequences as given in "junction_aa" field in the input dataset.

References:

Davidsen, K., Olson, B. J., DeWitt, W. S., III, Feng, J., Harkins, E., Bradley, P., & Matsen, F. A., IV. (2019).
Deep generative models for T cell receptor protein sequences. eLife, 8, e46935. https://doi.org/10.7554/eLife.46935


**Specification arguments:**

- locus (str): which locus the sequence come from, e.g., TRB

- beta (float): VAE hyperparameter that balanced the reconstruction loss and latent dimension regularization

- latent_dim (int): latent dimension of the VAE

- linear_nodes_count (int): in linear layers, how many nodes to use

- num_epochs (int): how many epochs to use for training

- batch_size (int): how many examples to consider at the same time

- j_gene_embed_dim (int or None): dimension of J gene embedding; if None, it defaults to the number of unique J genes in the training data

- v_gene_embed_dim (int or None): dimension of V gene embedding; if None, it defaults to the number of unique V genes in the training data

- cdr3_embed_dim (int or None): dimension of the cdr3 embedding; if None, it defaults to the size of the amino-acid alphabet (including padding)

- pretrains (int): how many times to attempt pretraining to initialize the weights and use warm-up for the beta hyperparameter before the main training process

- warmup_epochs (int): how many epochs to use for training where beta hyperparameter is linearly increased from 0 up to its max value; this is in addition to num_epochs set above

- patience (int): number of epochs to wait before the training is stopped when the loss is not improving

- iter_count_prob_estimation (int): how many iterations to use to estimate the log probability of the generated sequence (the more iterations, the better the estimated log probability)

- vocab (list): which letters (amino acids) are allowed - this is automatically filled for new models (no need to set)

- max_cdr3_len (int): what is the maximum cdr3 length - this is automatically filled for new models (no need to set)

- unique_v_genes (list): list of allowed V genes (this will be automatically filled from the dataset if not provided here manually)

- unique_j_genes (list): list of allowed J genes (this will be automatically filled from the dataset if not provided here manually)

- device (str): name of the device where to train the model (e.g., cpu)

- learning_rate (float): learning rate for the optimizer (default is 0.001)

- validation_split (float): what percentage of the data to use for validation (default is 0.1)

- seed (int): random seed for the model or None


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_vae:
                SimpleVAE:
                    locus: beta
                    beta: 0.75
                    latent_dim: 20
                    linear_nodes_count: 75
                    num_epochs: 5000
                    batch_size: 10000
                    j_gene_embed_dim: 13
                    v_gene_embed_dim: 30
                    cdr3_embed_dim: 21
                    pretrains: 10
                    warmup_epochs: 20
                    patience: 20
                    device: cpu



SoNNia
''''''''''''''''''''''''''''''''''''''''''''''''''''


SoNNia models the selection process of T and B cell receptor repertoires. It is based on the SoNNia Python package.
It supports SequenceDataset as input, but not RepertoireDataset.

Original publication:
Isacchini, G., Walczak, A. M., Mora, T., & Nourmohammad, A. (2021). Deep generative selection models of T and B
cell receptor repertoires with soNNia. Proceedings of the National Academy of Sciences, 118(14), e2023141118.
https://doi.org/10.1073/pnas.2023141118

**Specification arguments:**

- locus (str): The locus of the receptor chain.

- batch_size (int): number of sequences to use in each batch

- epochs (int): number of epochs to train the model

- deep (bool): whether to use a deep model

- include_joint_genes (bool)

- n_gen_seqs (int)

- custom_model_path (str): path for the custom OLGA model if used

- default_model_name (str): name of the default OLGA model if used

- seed (int): random seed for the model or None

- num_processes (int): number of processes to use for sequence generation (default: 4)


 **YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_sonnia_model:
                SoNNia:
                    batch_size: 1e4
                    epochs: 5
                    default_model_name: humanTRB
                    deep: False
                    include_joint_genes: True
                    n_gen_seqs: 100



**Dimensionality reduction methods**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Dimensionality reduction methods are algorithms which can be used to reduce the dimensionality
of encoded datasets, in order to uncover and analyze patterns present in the data.

These methods can be used in the :ref:`ExploratoryAnalysis` and :ref:`Clustering` instructions.


KernelPCA
''''''''''''''''''''''''''''''''''''''''''''''''''''


Principal component analysis (PCA) method which wraps scikit-learn's KernelPCA, allowing for non-linear dimensionality
reduction. Input arguments for the method are the
same as supported by scikit-learn (see `KernelPCA scikit-learn documentation
<https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html>`_ for details).

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_kernel_pca:
                KernelPCA:
                    # arguments as defined by scikit-learn
                    n_components: 5
                    kernel: rbf



PCA
''''''''''''''''''''''''''''''''''''''''''''''''''''


Principal component analysis (PCA) method which wraps scikit-learn's PCA. Input arguments for the method are the
same as supported by scikit-learn (see `PCA scikit-learn documentation
<https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA>`_ for details).

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_pca:
                PCA:
                    # arguments as defined by scikit-learn
                    n_components: 2



TSNE
''''''''''''''''''''''''''''''''''''''''''''''''''''


t-distributed Stochastic Neighbor Embedding (t-SNE) method which wraps scikit-learn's TSNE. It can be useful for
visualizing high-dimensional data. Input arguments for the method are the
same as supported by scikit-learn (see `TSNE scikit-learn documentation
<https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE>`_ for details).


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_tsne:
                TSNE:
                    # arguments as defined by scikit-learn
                    n_components: 2
                    init: pca



UMAP
''''''''''''''''''''''''''''''''''''''''''''''''''''


Uniform manifold approximation and projection (UMAP) method which wraps umap-learn's UMAP. Input arguments for the method are the
same as supported by umap-learn (see `UMAP in the umap-learn documentation
<https://umap-learn.readthedocs.io/en/latest/>`_ for details).

Note that when providing the arguments for UMAP in the immuneML's specification, it is not possible to set
functions as input values (e.g., for the metric parameter, it has to be one of the predefined metrics available
in umap-learn).

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        ml_methods:
            my_umap:
                UMAP:
                    # arguments as defined by umap-learn
                    n_components: 2
                    n_neighbors: 15
                    metric: euclidean


