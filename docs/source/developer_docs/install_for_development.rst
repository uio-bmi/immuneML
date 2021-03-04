Set up immuneML for development
=========================================

Prerequisites:

- Create and activate a Python virtual environment with Python 3.8 (`set up a Python virtual environment <https://docs.python.org/3/library/venv.html>`_) or use conda instead with Python 3.8 (`set up conda virtual environment <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_).

- Under Windows, the Microsoft Visual C++ 14.0 or greater is required to install from requirements.txt.

- System requirements: at least 4GB of RAM memory and 15GB of disk space.

For development purposes, it is most convenient to clone the codebase using PyCharm. To set up the project in PyCharm, see
`the official JetBrains tutorial for creating a PyCharm project from an existing GitHub repository <https://www.jetbrains.com/help/pycharm/manage-projects-hosted-on-github.html>`_.
`PyCharm fails to create virtual environments with Python version 3.9 <https://github.com/coursera-dl/coursera-dl/issues/778>`_, using version 3.8 works.

Alternatively to using PyCharm, the following steps describe how to perform the process manually:

1. Create a directory where the code should be located and navigate to that directory.

2. Execute the command to clone the repository:

  .. code-block:: console

    git clone https://github.com/uio-bmi/immuneML.git

3. From the project folder (immuneML folder created when the repository was cloned
from GitHub), install the requirements from the requirements.txt file (this file can be found in the immuneML root folder):

  .. code-block:: console

    pip install -r requirements.txt -e .

  See also these troubleshooting issues:

  - :ref:`When installing all requirements from requirements.txt, there is afterward an error with yaml package (No module named yaml)`

  - :ref:`I get an error when installing PyTorch (could not find a version that satisfies the requirement torch)`

  If you want to install optional requirements (DeepRC or TCRdist), install the corresponding requirements files (some or all of them):

  .. code-block:: console

    pip install -r requirements_DeepRC.txt
    pip install -r requirements_TCRdist.txt

4. If not setting up the project in PyCharm, it might be necessary to manually add the root project folder to PYTHONPATH.
The syntax for Unix-based systems is the following:

  .. code-block:: console

    export PYTHONPATH=$PYTHONPATH:$(pwd)

To run a sample analysis, run from the terminal:

.. code-block:: console

  immune-ml-quickstart ./output_dir/

This will generate a synthetic dataset and run a simple machine machine learning analysis on the generated data.
The results folder will contain two sub-folders: one for the generated dataset (:code:`synthetic_dataset`) and one for the results of the machine
learning analysis (:code:`machine_learning_analysis`). The files named specs.yaml are the input files for immuneML that describe how to generate the dataset
and how to do the machine learning analysis. The index.html files can be used to navigate through all the results that were produced.
