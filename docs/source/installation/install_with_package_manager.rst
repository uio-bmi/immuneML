Install immuneML with a package manager
=========================================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML installation through a package manager
   :twitter:description: See tutorials on how to install immuneML with Conda or PyPI
   :twitter:image: https://docs.immuneml.uio.no/_images/receptor_classification_overview.png


This manual shows how to install immuneML using either conda or pip.


Install immuneML with conda
------------------------------

0. If a conda distribution is not already installed on the machine, see `the official conda installation documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

1. Once conda is working, create a directory for immuneML and navigate to the directory:

.. code-block:: console

  mkdir immuneML/
  cd immuneML/

2. Create a virtual environment using conda. immuneML has been tested extensively with Python versions 3.7 and 3.8, but not 3.9.
   To create a conda virtual environment with Python version 3.8, use:

.. code-block:: console

  conda create --prefix immuneml_env/ python=3.8

3. Activate the created environment:

.. code-block:: console

  conda activate immuneml_env/

4. To install immuneML using conda, run:

.. code-block:: console

  conda install -c bioconda immuneml

Install immuneML with pip
------------------------------

0. To install immuneML with pip, make sure to have Python version 3.7 or 3.8 installed. immuneML with later Python versions should also work, but it has
not been extensively tested. For more information on Python versions, see `the official Python website <https://www.python.org/>`_.

1. Create a virtual environment where immuneML will be installed. It is possible to install immuneML as a global package, but it is not
recommended as there might be conflicting versions of different packages. For more details, see `the official documentation on creating virtual environments with
Python <https://docs.python.org/3/library/venv.html>`_. To create an environment, run the following in the terminal (for Windows-specific commands,
see the virtual environment documentation linked above):

.. code-block:: console

  python3 -m venv ./immuneml_venv/

2. To activate the virtual environment on Mac/Linux, run the following command (for Windows, see the documentation in the previous step):

.. code-block:: console

  source ./immuneml_venv/bin/activate

3. To install `immuneML from PyPI <https://pypi.org/project/immuneML/>`_ in this virtual environment, run the following:

.. code-block:: console

  pip install immuneML

Alternatively, if you want to use the :ref:`TCRdistClassifier` ML method and corresponding :ref:`TCRdistMotifDiscovery` report, include the optional extra :code:`TCRdist`:

.. code-block:: console

  pip install immuneML[TCRdist]

See also this question under 'Troubleshooting': :ref:`I get an error when installing PyTorch (could not find a version that satisfies the requirement torch)`

Installing optional dependencies
----------------------------------

Optionally, if you want to use the :ref:`DeepRC` ML method and and corresponding :ref:`DeepRCMotifDiscovery` report, you also
have to install DeepRC dependencies using the :download:`requirements_DeepRC.txt <https://raw.githubusercontent.com/uio-bmi/immuneML/master/requirements_DeepRC.txt>` file.
Important note: DeepRC uses PyTorch functionalities that depend on GPU. Therefore, DeepRC does not work on a CPU.
To install the DeepRC dependencies, run:

.. code-block:: console

  pip install -r requirements_DeepRC.txt --no-dependencies

If you want to use the :ref:`CompAIRRDistance` or :ref:`CompAIRRSequenceAbundance` encoder, you have to install the C++ tool `CompAIRR <https://github.com/uio-bmi/compairr>`_.
The easiest way to do this is by cloning CompAIRR from GitHub and installing it using :code:`make` in the main folder:

.. code-block:: console

  git clone https://github.com/uio-bmi/compairr.git
  cd compairr
  make install

If such installation is unsuccessful (for example if you do not have the rights to install CompAIRR via make),
it is also possible to directly provide the path to a CompAIRR executable as a parameter
to :ref:`CompAIRRDistance` or :ref:`CompAIRRSequenceAbundance` encoder.



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
    --version           show program's version and exit

2. To quickly test out whether immuneML is able to run, try running the quickstart command:

.. code-block:: console

    immune-ml-quickstart ./quickstart_results/

This will generate a synthetic dataset and run a simple machine machine learning analysis on the generated data.
The results folder will contain two sub-folders: one for the generated dataset (:code:`synthetic_dataset`) and one for the results of the machine
learning analysis (:code:`machine_learning_analysis`). The files named specs.yaml are the input files for immuneML that describe how to generate the dataset
and how to do the machine learning analysis. The index.html files can be used to navigate through all the results that were produced.
