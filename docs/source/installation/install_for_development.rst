Set up immuneML for development
----------------------------------
Prerequisites:

- Python 3.7 or 3.8: it might work with other python versions (3.6), but might require additional packages to be manually installed (e.g., dataclasses package if running immuneML with Python 3.6). Alternatively, a custom python interpreter can be assigned to the virtual environment (in PyCharm, for development purposes, or in a conda environment).

- Optionally R 3.6.x with libraries Rmisc and readr and library ggexp (which cannot be installed directly with conda, but can be installed with devtool library from `the GitHub repository <https://github.com/keshav-motwani/ggexp>`_). These libraries are necessary to generate  certain reports (SequenceAssociationLikelihood, FeatureValueBarplot, FeatureValueDistplot, SequencingDepthOverview, DensityHeatmap, FeatureHeatmap, SimilarityHeatmap).

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
