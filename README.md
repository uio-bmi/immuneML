# immuneML

![Python application](https://github.com/uio-bmi/immuneML/actions/workflows/python-app.yml/badge.svg?branch=master)
![Docker](https://github.com/uio-bmi/immuneML/actions/workflows/docker-publish.yml/badge.svg?branch=master)
![PyPI](https://github.com/uio-bmi/immuneML/actions/workflows/publish-to-pypi.yml/badge.svg?branch=master)
[![](https://img.shields.io/static/v1?label=AIRR-C%20sw-tools%20v1&message=compliant&color=008AFF&labelColor=000000&style=plastic)](https://docs.airr-community.org/en/stable/swtools/airr_swtools_standard.html)

immuneML is a platform for machine learning-based analysis and 
classification of adaptive immune receptors and repertoires (AIRR).

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
- Galaxy web interface: https://galaxy.immuneml.uiocloud.no



## Installation


We recommend installing immuneML inside a virtual environment.
immuneML uses **Python 3.9 or later**. If using immuneML simulation, Python 3.11 or later is recommended.
immuneML can be [installed directly using a package manager](<https://docs.immuneml.uio.no/latest/installation/install_with_package_manager.html#>) such as pip or conda,
or [set up via docker](<https://docs.immuneml.uio.no/latest/installation/installation_docker.html>).

#### Quick installation (immuneML essentials):

```bash
python3 -m venv ./immuneml_venv/
source ./immuneml_venv/bin/activate
pip install wheel
pip install immune-ml
```

or

```bash
conda create --prefix immuneml_env/ python=3.11
conda activate immuneml_env/
conda install -c bioconda immuneml
```

#### Detailed installation (immuneML extras):

Please check the documentation for more detailed instructions or [how to install optional dependencies](<https://docs.immuneml.uio.no/latest/installation/install_with_package_manager.html#installing-optional-dependencies>).

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
The files named `specs.yaml` are the input files for immuneML that describe how to generate 
the dataset and how to do the machine learning analysis. The `index.html` files can be used 
to navigate through all the results that were produced.

## Usage 

### Quickstart

The quickest way to familiarize yourself with immuneML usage is to follow
one of the [Quickstart tutorials](https://docs.immuneml.uio.no/quickstart.html).
These tutorials provide a step-by-step guide on how to use immuneML for a 
simple machine learning analysis on an adaptive immune receptor repertoire (AIRR) dataset,
using either the command line tool or the [Galaxy web interface](https://galaxy.immuneml.uiocloud.no). 


### Overview of immuneML analyses

The figure below shows an overview of immuneML usage. 
All parameters for an immuneML analysis are defined in the YAML specification file.
In this file, the settings of the analysis components are defined (also known as `definitions`, 
shown in different colors in the figure). 
Additionally, the YAML file describes one or more `instructions`, which are workflows that are
applied to the defined analysis components. 
See also: [documentation of the YAML specification](https://docs.immuneml.uio.no/latest/yaml_specs/how_to_specify_an_analysis_with_yaml.html).

Each instruction produces different types of results, including trained ML models, 
ML model predictions on a given dataset, plots or other reports describing the 
dataset or trained models, or synthetic/simulated datasets. 
These results can be navigated through the summary HTML file. 
See also: [tutorials for specific immuneML use cases](https://docs.immuneml.uio.no/latest/tutorials.html#).


![image info](https://docs.immuneml.uio.no/latest/_images/definitions_instructions_overview.png)


### Command line usage 

The `immune-ml` command takes only two parameters: the YAML specification file and a result path. 
An example is given here:

```bash
immune-ml path/to/specification.yaml result/folder/path/
```

### Results of an immuneML run

For each instruction specified in the YAML specification file, a subfolder is created in the 
`result/folder/path`. Each subfolder will contain:
- An `index.html` file which shows an overview of the results produced by that instruction. Inspecting the results of an immuneML analysis typically starts here. 
- A copy of the used YAML specification (`full_specification.yaml`) with all default parameters explicitly set.
- A log file (`log.txt`).
- A folder containing the imported dataset(s) in immuneML format.
- A folder containing all raw results produced by the instruction.

## Support

We will prioritize fixing important bugs, and try to answer any questions as soon as possible.
Please note we are only 2 people maintaining the platform with occasional absences.

When experiencing an issue, please take the following steps:

1. **Make sure the latest version of immuneML is installed.** immuneML is under constant development, and the issue you experience may already be resolved in the latest version of the platform.

2. Check the ['troubleshooting' page](<https://docs.immuneml.uio.no/latest/troubleshooting.html>) in the immuneML documentation. Any known issues and their solutions are already described there.

3. If you are still experience a problem and suspect a bug in immuneMl, you can [report an issue on GitHub](https://github.com/uio-bmi/immuneML/issues). Please make sure to include the following information:
    - The YAML specification you tried to run.
    - The full output log file (log.txt).
    - A list of dependency versions (can be retrieved with pip list or conda list).
    - We primarily test immuneML using Unix-based operating systems, please make sure to mention it if you're using Windows.
    - We will be able to help you fastest if you can also provide a small reproducible example, such as a very small dataset for which your run fails. 

  
If this does not answer your question, you can contact us via:
- Twitter [`@immuneml`](https://twitter.com/immuneml)
- Email [`contact@immuneml.uio.no`](mailto:contact@immuneml.uio.no)



# Citing immuneML

If you are using immuneML in any published work, please cite:

Pavlović, M., Scheffer, L., Motwani, K. et al. The immuneML ecosystem for machine learning analysis of adaptive immune 
receptor repertoires. Nat Mach Intell 3, 936–944 (2021). https://doi.org/10.1038/s42256-021-00413-z



<hr>


© Copyright 2021-2022, Milena Pavlovic, Lonneke Scheffer, Keshav Motwani, Victor Greiff, Geir Kjetil Sandve


