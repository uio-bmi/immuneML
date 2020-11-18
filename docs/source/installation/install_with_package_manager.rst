Install immuneML with a package manager
=========================================

This manual shows how to install immuneML with `Anaconda <https://docs.anaconda.com/anaconda/install/>`_ (tested with version 4.8.3).

While the immuneML core functionalities do not depend on R, R dependencies are necessary to generate certain plots (see: :ref:`When should I install immuneML with R dependencies?`).
Installing immuneML with R dependencies takes a few more steps. Therefore, both tutorials are shown separately below.


Install immuneML without R dependencies
---------------------------------------

1. Create a directory for immuneML and navigate to the directory:

.. code-block:: console

  mkdir immuneML/
  cd immuneML/

2. Create a virtual environment using conda:

.. code-block:: console

  conda create --prefix immuneml_env/ python=3.7.6

3. Activate the created environment:

.. code-block:: console

  conda activate immuneml_env/

4. Install basic immuneML including Python dependencies from GitHub using pip:

.. code-block:: console

  pip install git+https://github.com/uio-bmi/immuneML

Alternatively, if you want to install immuneML including optional extras (:code:`DeepRC`, :code:`TCRDist`), use:

.. code-block:: console

  pip install git+https://github.com/uio-bmi/immuneML#egg=immuneML[DeepRC,TCRDist]

Installing DeepRC and TCRDist dependencies is necessary to use the :ref:`DeepRC` and :ref:`TCRDISTClassifier` ML methods.
It is also possible to specify a subset of extras, for example, include only :code:`DeepRC`.

5. To validate the installation, run:

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



Install immuneML with R dependencies
---------------------------------------

1. Create a directory for immuneML and navigate to the directory:

.. code-block:: console

  mkdir immuneML/
  cd immuneML/

2. Create a virtual environment using conda, and install dependencies using the `environment.yaml <https://drive.google.com/file/d/1Vc7ivHL4z4l3KAyDX8qJ_Lsez_1nEb6e/view?usp=sharing>`_ file:

.. code-block:: console

  conda env create --prefix immuneml_env/ -f environment.yaml

3. Activate the created environment:

.. code-block:: console

  conda activate immuneml_env/

4. Install additional R dependencies from the script provided `here <https://drive.google.com/file/d/1C0m7bjG7OKfWNVQsgYkE-nXCdvD7mO08/view?usp=sharing>`_

.. code-block:: console

  sh install_immuneML_R_dependencies.sh

5. Install basic immuneML including Python and R dependencies from GitHub using pip:

.. code-block:: console

  pip install git+https://github.com/uio-bmi/immuneML#egg=immuneML[R_plots]

Alternatively, if you want to install immuneML including optional extras (:code:`DeepRC`, :code:`TCRDist`), use:

.. code-block:: console

  pip install git+https://github.com/uio-bmi/immuneML#egg=immuneML[R_plots,DeepRC,TCRDist]

Installing DeepRC and TCRDist dependencies is necessary to use the :ref:`DeepRC` and :ref:`TCRDISTClassifier` ML methods.

6. To validate the installation, run:

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

More information on conda environments (how to activate, deactivate environment) is available on `the conda site <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.


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

Alternatively, if you want to install immuneML including optional dependencies (:code:`R_plots`, :code:`DeepRC`, :code:`TCRDist`), use:

.. code-block:: console

  pip install git+https://github.com/uio-bmi/immuneML#egg=immuneML[R_plots,DeepRC,TCRDist]

Note: when including R_plots, make sure that R dependencies were installed using the steps described in :ref:`Install immuneML with R dependencies` steps 2 - 4.
