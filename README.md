# immuneML

immuneML is a software platform for machine learning analysis of immune receptor sequences.

It supports the analyses of experimental B- and T-cell receptor data,
as well as synthetic data for benchmarking purposes.

In ImmuneML, users can define flexible workflows supporting various
machine learning libraries (such as scikit-learn, PyTorch, Tensorflow, 
statsmodels), benchmarking of different approaches, numerous reports
of data characteristics, ML algorithms and their predictions, and
visualizations of results.

Additionally, users can extend the platform by defining their own data
 representations, ML models, reports and visualizations.

The platform is open-source with source code available on GitHub.

## Getting started

Steps to setup immuneML environment:

1. Make sure that your machine has python 3.8 installed (or install if necessary: https://www.python.org/downloads/). immuneML can also
work with python 3.7 and 3.6, but some additional packages may need to be manually installed in that case (e.g. dataclasses).
3. Create a virtual environment (as described here: https://docs.python.org/3/library/venv.html)
4. Install the package using pip:

    `pip install git+https://github.com/uio-bmi/ImmuneML`
    
To try out the platform, see the quickstart guide and the documentation.
