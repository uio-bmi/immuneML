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

1. Make sure that your machine has python 3.7 installed (or install if necessary: https://www.python.org/downloads/). immuneML can also
work with python 3.6, but some additional packages may need to be manually installed in that case (e.g. dataclasses).
2. Clone this repository, 
3. Create a virtual environment (as described here: https://docs.python.org/3/library/venv.html)
4. Install dependencies listed in requirements.txt located directly in the project folder (for instance using pip3):

   `pip3 install -r requirements.txt`
  
5. To run a sample analysis, from the project folder with activated virtual environment run:

   `python3 source/workflows/instructions/quickstart.py`
   
The sample analysis will simulate a repertoire dataset in which 50% of repertoires are celiac-specific, 
while the rest correspond to healthy patients. The quickstart will then generate a sample run specification, 
and run nested cross-validation to determine the optimal set of parameters to predict celiac status. 
The results of the analysis will be located in the quickstart folder under the project root folder, 
while the log will be printed to the standard output 
(in this case, the printed prediction accuracy should be around 0.5 as the repertoires are randomly generated).
