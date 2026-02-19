Install immuneML with a package manager
=========================================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML installation through a package manager
   :twitter:description: See tutorials on how to install immuneML with Conda or PyPI
   :twitter:image: https://docs.immuneml.uio.no/_images/receptor_classification_overview.png


This manual shows how to install immuneML using either conda or pip.

Install immuneML with pip
------------------------------

0. To install immuneML with pip, make sure to have Python version **3.9 or later** installed.

1. Create a virtual environment where immuneML will be installed. It is possible to install immuneML as a global
   package, but it is not recommended as there might be conflicting versions of different packages. For more details,
   see `the official documentation on creating virtual environments with Python <https://docs.python.org/3/library/venv.html>`_.
   To create an environment, run the following in the terminal (for Windows-specific commands, see the virtual
   environment documentation linked above):

.. code-block:: console

  python3 -m venv ./immuneml_venv/

2. To activate the virtual environment on Mac/Linux, run the following command (for Windows, see the documentation in the previous step):

.. code-block:: console

  source ./immuneml_venv/bin/activate

Note: when creating a python virtual environment, it will automatically use the same Python version as the environment
it was created in. To ensure that the preferred Python version (3.9 or later) is used, it is possible to instead make a conda
environment (see :ref:`Install immuneML with conda` steps 0-3) and proceed to install immuneML with pip inside the
conda environment.


3. If not already up-to-date, update pip:

.. code-block:: console

  python3 -m pip install --upgrade pip


4. To install `immuneML from PyPI <https://pypi.org/project/immuneML/>`_ in this virtual environment, run the following:

.. code-block:: console

  pip install immuneML



Install immuneML with conda
------------------------------

0. If a conda distribution is not already installed on the machine, see `the official conda installation documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

1. Once conda is working, create a directory for immuneML and navigate to the directory:

.. code-block:: console

  mkdir immuneML/
  cd immuneML/

2. Create a virtual environment using conda. immuneML should work with Python version 3.9 or later, and has been tested extensively with Python version 3.11.
   To create a conda virtual environment with Python version 3.11, use:

.. code-block:: console

  conda create --prefix immuneml_env/ python=3.11

3. Activate the created environment:

.. code-block:: console

  conda activate immuneml_env/

4. To install immuneML using conda, run:

.. code-block:: console

  conda install -c bioconda immuneml


Installing optional dependencies
----------------------------------

TCRDist
*******

If you want to use the :ref:`TCRdistClassifier` ML method and corresponding :ref:`TCRdistMotifDiscovery` report, you can include the optional extra :code:`TCRdist`:

.. code-block:: console

  pip install immuneML[TCRdist]

The TCRdist dependencies can also be installed manually using the :download:`requirements_TCRdist.txt <https://raw.githubusercontent.com/uio-bmi/immuneML/master/requirements_TCRdist.txt>` file:

.. code-block:: console

  pip install -r requirements_TCRdist.txt


DeepRC
******

Optionally, if you want to use the :ref:`DeepRC` ML method and and corresponding :ref:`DeepRCMotifDiscovery` report, you also
have to install DeepRC dependencies using the :download:`requirements_DeepRC.txt <https://raw.githubusercontent.com/uio-bmi/immuneML/master/requirements_DeepRC.txt>` file.
Important note: DeepRC uses PyTorch functionalities that depend on GPU. Therefore, DeepRC does not work on a CPU.
To install the DeepRC dependencies, run:

.. code-block:: console

  pip install -r requirements_DeepRC.txt --no-dependencies

See also this question under 'Troubleshooting': :ref:`I get an error when installing PyTorch (could not find a version that satisfies the requirement torch)`


Deep learning methods
************************

In order to use any of the supported deep learning models (KerasSequenceCNN or others), install DL optional dependencies:

.. code-block:: console

  pip install immuneML[DL]

Fisher's exact test
**********************

For using ProbabilisticBinaryClassifier or any of the abundance encoders (following Emerson et al. 2017 publication),
please install 'fisher' optional dependencies:

.. code-block:: console

  pip install immuneML[fisher]

Full immuneML installation
******************************

To install all optional dependencies and have access to the full set of immuneML features, use the following installation command:

.. code-block:: console

  pip install immuneML[all]

CompAIRR
********

If you want to use the :ref:`CompAIRRDistance` or :ref:`CompAIRRSequenceAbundance` encoder, you have to install the C++ tool `CompAIRR <https://github.com/uio-bmi/compairr>`_.
Furthermore, the :ref:`SimilarToPositiveSequence` encoder can be run both with and without CompAIRR, but the CompAIRR-based version is faster.

The easiest way to install CompAIRR is by cloning CompAIRR from GitHub and installing it using :code:`make` in the main folder:

.. code-block:: console

  git clone https://github.com/uio-bmi/compairr.git
  cd compairr
  make install

If such installation is unsuccessful (for example if you do not have the rights to install CompAIRR via make),
it is also possible to directly provide the path to a CompAIRR executable as a parameter
to :ref:`CompAIRRDistance` or :ref:`CompAIRRSequenceAbundance` encoder.



Testing immuneML
-----------------

To validate the installation, run:

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


.. include:: ./run_quickstart.rst