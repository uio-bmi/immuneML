Installing immuneML
===================

.. toctree::
   :maxdepth: 2

To get started with immuneML, there are three options:

1. Installing immuneML with a package manager and using it locally (recommended),

2. Cloning the codebase from GitHub repository and running it directly from source code (for development purposes),

3. Using immuneML from Galaxy (no programming experience required, the functionality is available through a web interface, see :ref:`immuneML & Galaxy`).

In this section, it will be described how to set up the first two options. Once immuneML has been set up, you can look into :ref:`Quickstart` or
look into which analyses you can run with immuneML (see :ref:`Tutorials`).

Install immuneML with a package manager
---------------------------------------

Steps to install immuneML with Anaconda (tested with version 4.8.3):

1. Create a directory for immuneML and navigate to the directory:

.. code-block:: console

  mkdir immuneML_test/
  cd immuneML_test/

2. Create a virtual environment using conda, and install dependencies using `environment.yaml <https://drive.google.com/file/d/1Vc7ivHL4z4l3KAyDX8qJ_Lsez_1nEb6e/view?usp=sharing>`_ file:

.. code-block:: console

  conda env create --prefix immuneml_env/ -f environment.yaml

3. Activate created environment:

.. code-block:: console

  conda activate immuneml_env/

4. Optionally, install additional R dependencies from the script provided `here <https://drive.google.com/file/d/1C0m7bjG7OKfWNVQsgYkE-nXCdvD7mO08/view?usp=sharing>`_
(note that the immuneML core functionality does not depend on R, it is only necessary to generate certain reports):

.. code-block:: console

  sh install_immuneML_R_dependencies.sh

5. Install immuneML including Python dependencies from GitHub using pip:

.. code-block:: console

  pip install git+https://github.com/uio-bmi/immuneML

If you want to install immuneML including all R plots, use instead:

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

If you want to install immuneML including all R plots, use instead:

  pip install git+https://github.com/uio-bmi/immuneML#egg=immuneML[R_plots]


Clone the codebase from GitHub
-------------------------------
Prerequisites:

- Python 3.7.6: it might work with other python versions (3.6), but might require additional packages to be manually installed (e.g., dataclasses package if running immuneML with Python 3.6). Alternatively, a custom python interpreter can be assigned to the virtual environment (in PyCharm, for development purposes, or in a conda environment).

- Optionally R 3.6.x with libraries Rmisc and readr and library ggexp (which cannot be installed directly with conda, but can be installed with
devtool library from `the GitHub repository <https://github.com/keshav-motwani/ggexp>`_). These libraries are necessary to generate  certain reports (SequenceAssociationLikelihood,
FeatureValueBarplot, FeatureValueDistplot, SequencingDepthOverview, DensityHeatmap, FeatureHeatmap, SimilarityHeatmap).

Note: for development purposes, it is much more convenient to clone the codebase using PyCharm. To set up the project in PyCharm, see
`the official JetBrains tutorial for creating a PyCharm project from an existing GitHub repository <https://www.jetbrains.com/help/pycharm/manage-projects-hosted-on-github.html>`_.
Alternatively, the following 5 steps describe how to perform the process manually.

Steps:

1. Create a directory where the code should be located and navigate to that directory.

2. Execute the command to clone the repository:

.. code-block:: console

  git clone https://github.com/uio-bmi/immuneML.git

3. Create and activate a virtual environment as described here
https://docs.python.org/3/library/venv.html (python virtual environment)
or use conda instead with python 3.7.

4. From the project folder (immuneML folder created when the repository was cloned
from GitHub), install the requirements from requirements.txt file:

.. code-block:: console

  pip install -r requirements.txt

To run a sample analysis, from the project folder run:

.. code-block:: console

  python3 source/workflows/instructions/quickstart.py
