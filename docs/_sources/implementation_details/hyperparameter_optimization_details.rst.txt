Hyperparameter optimization details
===================================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML: hyperparameter optimization
   :twitter:description: See in a greater detail how immuneML does hyperparameter optimization through nested cross-validation.
   :twitter:image: https://docs.immuneml.uio.no/_images/receptor_classification_overview.png



The flow of the hyperparameter optimization is shown in the figure below, along with the output that is generated and reports executed during the particular step:

.. figure:: ../_static/images/hp_optmization_with_outputs.png
  :width: 70%

  Execution flow of the TrainMLModel along with the information on data and reports generated at each step

The code is split into `TrainMLModelInstruction`, `HPAssessment`, `HPSelection`, `HPUtil` and `HPReports` classes. All code related to hyperparameter
optimization is located in the hyperparameter_optimization package, except TrainMLModelInstruction which is located in the instructions package.

The parameters used to define the hyperparameter optimization, such as dataset, how to split the data, which batch size to use,
along with intermediary results, such as split data and trained ML models are all kept in the `TrainMLModelState` class instance.
This data class is also passed to specified hyperparameter reports. More details on the TrainMLModelState are shown in the figure below.

.. figure:: ../_static/images/state_hierarchy_hp_optimization.png
  :width: 70%

  Hyperparameter optimization state hierarchy of classes along with information each of these classes includes
