Install immuneML with a package manager
=========================================

Steps to install immuneML with Anaconda (tested with version 4.8.3):

1. Create a directory for immuneML and navigate to the directory:

.. code-block:: console

  mkdir immuneML/
  cd immuneML/

2. Create a virtual environment using conda, and install dependencies using `environment.yaml <https://drive.google.com/file/d/1Vc7ivHL4z4l3KAyDX8qJ_Lsez_1nEb6e/view?usp=sharing>`_ file:

.. code-block:: console

  conda env create --prefix immuneml_env/ -f environment.yaml

3. Activate created environment:

.. code-block:: console

  conda activate immuneml_env/

4. Optionally, install additional R dependencies from the script provided `here <https://drive.google.com/file/d/1C0m7bjG7OKfWNVQsgYkE-nXCdvD7mO08/view?usp=sharing>`_
(note that the immuneML core functionality does not depend on R, it is only necessary to generate certain reports. See: :ref:`When should I install immuneML with R dependencies?`):

.. code-block:: console

  sh install_immuneML_R_dependencies.sh

5. Install immuneML including Python dependencies from GitHub using pip:

.. code-block:: console

  pip install git+https://github.com/uio-bmi/immuneML

Alternatively, if you want to install immuneML including all R plots, use:

.. code-block:: console

  pip install git+https://github.com/uio-bmi/immuneML#egg=immuneML[R_plots]

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

Alternatively, if you want to install immuneML including all R plots, use:

.. code-block:: console

  pip install git+https://github.com/uio-bmi/immuneML#egg=immuneML[R_plots]