Install immuneML with a package manager
=========================================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML installation through a package manager
   :twitter:description: See tutorials on how to install immuneML with PyPI
   :twitter:image: https://docs.immuneml.uio.no/_images/receptor_classification_overview.png


This manual shows how to `install immuneML with pip <https://pypi.org/project/immuneML/>`_ in an
`Anaconda <https://docs.anaconda.com/anaconda/install/>`_ virtual environment.


Install immuneML
-----------------

1. Create a directory for immuneML and navigate to the directory:

.. code-block:: console

  mkdir immuneML/
  cd immuneML/

2. Create a virtual environment using conda. immuneML has been tested extensively with Python versions 3.7 and 3.8, but not 3.9.
   immuneML depends on PyTorch, which fails to install with Python 3.9.
   To create a conda virtual environment with Python version 3.8, use:

.. code-block:: console

  conda create --prefix immuneml_env/ python=3.8

3. Activate the created environment:

.. code-block:: console

  conda activate immuneml_env/

4. Install basic immuneML including Python dependencies using pip:

.. code-block:: console

  pip install immuneML

Alternatively, if you want to use the :ref:`TCRdistClassifier` ML method and corresponding :ref:`TCRdistMotifDiscovery` report, include the optional extra :code:`TCRdist`:

.. code-block:: console

  pip install immuneML[TCRdist]

See also this question under 'Troubleshooting': :ref:`I get an error when installing PyTorch (could not find a version that satisfies the requirement torch)`

5. Optionally, if you want to use the :ref:`DeepRC` ML method and and corresponding :ref:`DeepRCMotifDiscovery` report, you also
have to install DeepRC dependencies using the :download:`requirements_DeepRC.txt <https://raw.githubusercontent.com/uio-bmi/immuneML/master/requirements_DeepRC.txt>` file.
Important note: DeepRC uses PyTorch functionalities that depend on GPU. Therefore, DeepRC does not work on a CPU.
To install the DeepRC dependencies, run:

.. code-block:: console

  pip install -r requirements_DeepRC.txt --no-dependencies

6. Optionally, if you want to use the :ref:`CompAIRRDistanceEncoder` or :ref:`CompAIRRSequenceAbundanceEncoder`, you have to install the C++ tool `CompAIRR <https://github.com/uio-bmi/compairr>`_.
The easiest way to do this is by cloning CompAIRR from GitHub and installing it using :code:`make` in the main folder:

.. code-block:: console

  git clone https://github.com/uio-bmi/compairr.git
  cd compairr
  make install





Testing immuneML
-----------------

1. To validate the installation, run:

.. code-block:: console

  immune-ml -h

The output should look like this:

.. code-block:: console

  usage: immune-ml [-h] [--tool TOOL] specification_path result_path

  immuneML command line tool

  positional arguments:
    specification_path  Path to specification YAML file. Always used to define
                        the analysis.
    result_path         Output directory path.

  optional arguments:
    -h, --help          show this help message and exit
    --tool TOOL         Name of the tool which calls immuneML. This name will be
                        used to invoke appropriate API call, which will then do
                        additional work in tool-dependent way before running
                        standard immuneML.

2. To quickly test out whether immuneML is able to run, try running the quickstart command:

.. code-block:: console

    immune-ml-quickstart ./quickstart_results/

This will generate a synthetic dataset and run a simple machine machine learning analysis on the generated data.
The results folder will contain two sub-folders: one for the generated dataset (:code:`synthetic_dataset`) and one for the results of the machine
learning analysis (:code:`machine_learning_analysis`). The files named specs.yaml are the input files for immuneML that describe how to generate the dataset
and how to do the machine learning analysis. The index.html files can be used to navigate through all the results that were produced.
