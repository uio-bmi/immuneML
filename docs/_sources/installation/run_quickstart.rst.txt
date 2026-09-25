To quickly test out whether immuneML is able to run, try running the quickstart command:

.. code-block:: console

    immune-ml-quickstart ./quickstart_results/

This will generate a synthetic dataset and run a simple machine machine learning analysis on the generated data.
The results folder will contain two sub-folders: one for the generated dataset (:code:`synthetic_dataset`) and one for the results of the machine
learning analysis (:code:`machine_learning_analysis`). The files named specs.yaml are the input files for immuneML that describe how to generate the dataset
and how to do the machine learning analysis. The index.html files can be used to navigate through all the results that were produced.
