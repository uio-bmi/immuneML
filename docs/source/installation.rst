Installing immuneML
===================

.. toctree::
   :maxdepth: 2

There are four options to use immuneML:

1. Installing immuneML with a package manager and using it locally (recommended),

2. Cloning the codebase from GitHub repository and running it directly from source code (for development purposes),

3. Using immuneML from the immunoHub server (for local users),

4. Using immuneML from Galaxy (no programming experience required, the functionality is available through web interface, see :ref:`immuneML & Galaxy`).

In this section, it will be described how to set up the first two options. Once this is set up, you can look into :ref:`Quickstart` or
look into which analyses you can run with immuneML (see :ref:`Tutorials`).

Install immuneML with a package manager
---------------------------------------

Steps to install immuneML with Anaconda (version 4.8.3):

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

4. Install additional R dependencies from the script provided `here <https://drive.google.com/file/d/1C0m7bjG7OKfWNVQsgYkE-nXCdvD7mO08/view?usp=sharing>`_:

.. code-block:: console

  sh install_immuneML_R_dependencies.sh

5. Navigate to immuneML_test directory and clone the immuneML repository from GitHub (it will create ImmuneML folder):

.. code-block:: console

  git clone https://github.com/uio-bmi/ImmuneML.git

6. Install immuneML (it will also automatically install all dependencies):

.. code-block:: console

  python3 -m pip install ./ImmuneML/

7. To validate installation, run:

.. code-block:: console

  immune-ml -h

The output should describe command line arguments and give basic information about the package.

More information on conda environments (how to activate, deactivate environment) is available on `conda site <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

How to update immuneML if it was already installed
--------------------------------------------------

To check the existing version, activate the virtual environment where immuneML is installed (step 3 in the previous tutorial) and run the following command:

.. code-block:: console

  pip show immune-ml

The output of this command includes package name, version and other information.

To update the existing installation (obtained as described before):

1. Navigate to the immuneML repository directory

2. Pull the latest version of the repository from GitHub:

.. code-block:: console

  git pull origin master

3. Activate the virtual environment you created:

.. code-block:: console

  conda activate ../immuneml_env/

4. Install the new version of immuneML using pip from the repository directory:

.. code-block:: console

  python3 -m pip install ./


Clone the codebase from GitHub
-------------------------------
Prerequisites:

- Python 3.7.6: it might work with other python versions (3.6), but might require additional packages to be manually installed (dataclasses package if running immuneML with Python 3.6). Alternatively, a custom python interpreter can be assigned to the virtual environment (in PyCharm, for development purposes, or in a conda environment).

- R 3.6.3 with libraries ggplot2, here, ggsci, viridis, devtools, BiocManager, Rmisc, patchwork. Additional libraries (which cannot be installed directly with conda) are ComplexHeatmap (can be installed from BiocManager) and ggexp (can be installed with devtool library from the GitHub repository). These libraries are necessary to generate reports. Other functionalities in immuneML (training a model, preprocessing, encoding, importing or exporting the data) are not dependent on R.

Note: for development purposes, it is much more convenient to clone the codebase using PyCharm. To set up the project in PyCharm, see the official JetBrains tutorial for creating a PyCharm project from an existing GitHub repository. Alternatively, the following 5 steps describe how to do the process manually.

Steps:

1. Create a directory where the code should be located and navigate to that directory.

2. Execute the command to clone the repository:

.. code-block:: console

  git clone https://github.com/uio-bmi/ImmuneML.git

3. Create and activate a virtual environment as described here
https://docs.python.org/3/library/venv.html (python virtual environment)
or use conda instead with python 3.7.

4. From the project folder (ImmuneML folder created when the repository was cloned
from GitHub), install the requirements from requirements.txt file:

.. code-block:: console

  pip3 install -r requirements.txt

To run a sample analysis, from the project folder run:

.. code-block:: console

  python3 source/workflows/instructions/quickstart.py
