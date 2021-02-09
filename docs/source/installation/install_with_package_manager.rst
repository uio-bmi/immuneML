Install immuneML with a package manager
=========================================

This manual shows how to install immuneML with `Anaconda <https://docs.anaconda.com/anaconda/install/>`_ (tested with version 4.8.3).


Install immuneML
-----------------

1. Create a directory for immuneML and navigate to the directory:

.. code-block:: console

  mkdir immuneML/
  cd immuneML/

2. Create a virtual environment using conda. immuneML has been tested extensively with Python versions 3.7 and 3.8, but not 3.9. To create a conda virtual environment with Python version 3.8, use:

.. code-block:: console

  conda create --prefix immuneml_env/ python=3.8

3. Activate the created environment:

.. code-block:: console

  conda activate immuneml_env/

4. Install basic immuneML including Python dependencies from GitHub using pip:

.. code-block:: console

  pip install git+https://github.com/uio-bmi/immuneML

Alternatively, if you want to install immuneML including optional extras (:code:`DeepRC`, :code:`TCRDist`), use:

.. code-block:: console

  pip install git+https://github.com/uio-bmi/immuneML#egg=immuneML[DeepRC,TCRDist]

Installing DeepRC and TCRDist dependencies is necessary to use the :ref:`DeepRC` and :ref:`TCRDISTClassifier` ML methods, and corresponding :ref:`DeepRCMotifDiscovery` and :ref:`TCRDistMotifDiscovery` reports.
It is also possible to specify a subset of extras, for example, include only :code:`DeepRC`.

See also this FAQ: :ref:`I get an error when installing PyTorch (could not find a version that satisfies the requirement torch)`


How to update immuneML if it was already installed
--------------------------------------------------

To check the existing version of immuneML, activate the virtual environment where immuneML is installed (step 3 in the previous tutorial) and run the following command:

.. code-block:: console

  pip show immune-ml

If immuneML is already installed, the output of this command includes package name, version and other information.

To update the existing installation (obtained as described before):

1. Activate the virtual environment you created:

.. code-block:: console

  conda activate immuneml_env/

2. Install the new version of immuneML using pip:

.. code-block:: console

  pip install git+https://github.com/uio-bmi/immuneML


Alternatively, if you want to install immuneML including :code:`all` optional extras, use:

.. code-block:: console

  pip install git+https://github.com/uio-bmi/immuneML#egg=immuneML[all]

Or specify the specific extras you want to install (choose from :code:`DeepRC`, :code:`TCRDist`).
Note that specifying all these extras is equivalent to specifying :code:`all`:

.. code-block:: console

  pip install git+https://github.com/uio-bmi/immuneML#egg=immuneML[DeepRC,TCRDist]


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
The results folder will contain two sub-folders: one for the generated dataset and one for the results of the machine
learning analysis. The files named specs.yaml are the input files for immuneML that describe the above-mentioned
analyses. The index.html files can be used to navigate through all the results that were produced.