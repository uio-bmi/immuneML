# immuneML

![Python application](https://github.com/uio-bmi/immuneML/workflows/Python%20application/badge.svg?branch=master)
![Docker](https://github.com/uio-bmi/immuneML/workflows/Docker/badge.svg?branch=master)

immuneML is a software platform for machine learning analysis of immune receptor sequences.

It supports the analyses of experimental B- and T-cell receptor data,
as well as synthetic data for benchmarking purposes.

In immuneML, users can define flexible workflows supporting different
machine learning libraries (such as scikit-learn or PyTorch), benchmarking of different approaches, numerous reports
of data characteristics, ML algorithms and their predictions, and
visualizations of results.

Additionally, users can extend the platform by defining their own data 
representations, ML models, reports and visualizations.


Useful links:
- Main website: https://immuneml.uio.no
- Documentation: https://docs.immuneml.uio.no
- Galaxy web interface: https://galaxy.immuneml.uio.no



## Installation

immuneML can be installed directly [using pip](<https://pypi.org/project/immuneML/>).
immuneML uses Python 3.7 or 3.8, we recommend installing immuneML inside a virtual environment 
with one of these Python versions. 

For more detailed instructions (virtual environment, troubleshooting, Docker, developer installation), please see the [installation documentation](https://docs.immuneml.uio.no/installation/install_with_package_manager.html).

### Installation using pip


To install the immuneML core package, run:

```bash
pip install immuneML
```

Alternatively, to use the TCRdistClassifier ML method and corresponding TCRdistMotifDiscovery report, install immuneML with the optional TCRdist extra:

```bash
pip install immuneML[TCRdist]
```

Optionally, if you want to use the DeepRC ML method and and corresponding DeepRCMotifDiscovery report, you also
have to install DeepRC dependencies using the [requirements_DeepRC.txt](https://raw.githubusercontent.com/uio-bmi/immuneML/master/requirements_DeepRC.txt) file.
Important note: DeepRC uses PyTorch functionalities that depend on GPU. Therefore, DeepRC does not work on a CPU.
To install the DeepRC dependencies, run:

```bash
pip install -r requirements_DeepRC.txt
```

### Validating the installation

To validate the installation, run:

```bash
immune-ml -h
```

This should display a help message explaining immuneML usage.

To quickly test out whether immuneML is able to run, try running the quickstart command:

```bash
immune-ml-quickstart ./quickstart_results/
```

This will generate a synthetic dataset and run a simple machine machine learning analysis 
on the generated data. The results folder will contain two sub-folders: one for the generated dataset (`synthetic_dataset`) 
and one for the results of the machine learning analysis (`machine_learning_analysis`). 
The files named specs.yaml are the input files for immuneML that describe how to generate 
the dataset and how to do the machine learning analysis. The index.html files can be used 
to navigate through all the results that were produced.

## Usage 
todo 





## Requirements

- [Python 3.7 or 3.8](https://www.python.org/)
- Python packages:
   - [airr](https://pypi.org/project/airr/) (1 or higher)
   - [dill](https://pypi.org/project/dill/) (0.3 or higher)
   - [editdistance](https://pypi.org/project/editdistance/) (0.5.3 or higher)
   - [fishersapi](https://pypi.org/project/fishersapi/)
   - [gensim](https://pypi.org/project/gensim/) (3.8 or higher)
   - [h5py](https://www.h5py.org/) (2.10.0 or lower when using the optional DeepRC dependency)
   - [logomaker](https://pypi.org/project/logomaker/) (0.8 or higher)
   - [matplotlib](https://matplotlib.org) (3.1 or higher)
   - [matplotlib-venn](https://pypi.org/project/matplotlib-venn/) (0.11 or higher)
   - [numpy](https://www.numpy.org/) (1.18 or higher)
   - [pandas](https://pandas.pydata.org/) (1 or higher)
   - [plotly](https://plotly.com/python/) (4 or higher)
   - [pystache](https://pypi.org/project/pystache/) (0.5.4)
   - [Pytorch](https://pytorch.org/) (1.5.1 or higher)
   - [PyYAML](https://pyyaml.org) (5.3 or higher)
   - [regex](https://pypi.org/project/regex/) 
   - [requests](https://requests.readthedocs.io/) (2.21 or higher)
   - [scikit-learn](https://scikit-learn.org/) (0.23 or higher)
   - [scipy](https://www.scipy.org)
   - [tensorboard](https://www.tensorflow.org/tensorboard) (1.14.0)
   - [tqdm](https://tqdm.github.io/) (0.24 or higher)
   - [tzlocal](https://pypi.org/project/tzlocal/) 
- Optional dependencies when using DeepRC:
   - [DeepRC](https://github.com/ml-jku/DeepRC)
   - [widis-lstm-tools](https://github.com/widmi/widis-lstm-tools)
- Optional dependencies when using TCRdist:
   - [parasail](https://pypi.org/project/parasail/) (1.2)
   - [tcrdist3](https://github.com/kmayerb/tcrdist3) (0.1.6 or higher)


<hr>


© Copyright 2021, Milena Pavlovic, Lonneke Scheffer, Keshav Motwani, Victor Greiff, Geir Kjetil Sandve


