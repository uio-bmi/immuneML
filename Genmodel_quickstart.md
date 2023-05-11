# immuneML

![Docker](https://github.com/uio-bmi/immuneML/workflows/Docker/badge.svg?branch=master)
[![](https://img.shields.io/static/v1?label=AIRR-C%20sw-tools%20v1&message=compliant&color=008AFF&labelColor=000000&style=plastic)](https://docs.airr-community.org/en/stable/swtools/airr_swtools_standard.html)

immuneML is a platform for machine learning-based analysis and 
classification of adaptive immune receptors and repertoires (AIRR).

This branch of immuneML focuses on Generative Models and their use within immunology.
Following is an introduciton to the Generative Models that have been included and how 
to utilize them.

In order to utilize the generative models download this branch of immuneML, either downloading as a zip, or cloning the 
repository.


```bash
git clone https://github.com/uio-bmi/immuneML
```

Then checkout to the correct branch:

```bash
git checkout GenerativeModelsWUnsupervisedLearning
```

## Installation
When using this version of immuneML you need to install extra requirements to run the quickstart, all of these requirements can be found in the requirements.txt file.
In order to do this, run this command:

```bash
pip install -r requirements.txt
```

## Usage 

immuneML as a whole has many unique usages, most of which is not touched upon within this documentation. However, if 
you wish to know more about these see the README.md in the master branch: https://github.com/uio-bmi/immuneML

There are two main usages of Generative Models in immuneML, training and generation.
Both generates as many sequences as specified, or 100 if not specified.
The difference in the two is one requires a dataset in order to train the model first, while the other can take a finished
model as input.

### Quickstart

Within the Generative Models branch are a few files and directories, not found elsewhere in immuneML which can be
used to run a quickstart.<br/>
There are six YAML specifications, two for each type of generative model implemented, one for training and one for generation.
In order to execute the YAML files for loading, models first need to be trained and then referenced in the yaml.

The files generative_LSTM, generative_VAE, and generative_PWM are to be run for training, and the files with the added _load are used for loading existing models.
Within the training yamls there is little room for variation. They only work using the given encoding, and using repertoires.
The LSTM can be modified using the optional parameters of rnn_units and epochs. Moreover, every generative model can produce a requested amount of sequences
using the amount parameter in the instruction part of the yaml. If left unspecified, 100 is set as standard.

#### generative_PWM.yml
```yaml
definitions:
  datasets:
    d1:
      format: Generic
      params:
        path: datasets/sequences_under_30.tsv
        is_repertoire: False
        paired: False
        region_type: FULL_SEQUENCE
  encodings:
    e1: OneHot
  ml_methods:
    G1: PWM
  reports:
    GeneratorReport: GeneratorReport
instructions:
  machine_learning_instruction:
    type: GenerativeModel
    generators:
      generator_1:
        encoding: e1
        dataset: d1
        ml_method: G1
        report: GeneratorReport
```

Furthemore, the specification for only generation, using an existing model looks like this.
#### generative_load_quickstart.yml
```yaml

definitions:
  ml_methods:
    G1:
      LSTM:
        cores_for_training: 2
        amount: 200
  reports:
    GeneratorReport: GeneratorReportLSTM
instructions:
  machine_learning_instruction:
    type: GenerativeModelLoad
    generators:
      generator_1:
        ml_method: G1
        report: GeneratorReport
        path: LSTM_out/machine_learning_instruction/analysis_generator_1
```

#### Report

So far there are two different reports that can be used on the genrative models, GeneratorReport and NeuralNetGeneratorReport.
GeneratorReport can be used on all generative models but relays little information. While NeuralNetGeneratorReport is tailored 
for the generative models that produce loss values. These reports can be found in the output directory specified when running the
program.

### Overview of input, analyses and results

For a detailed explanation of the YAML specification file, see the tutorial [How to specify an analysis with YAML](https://docs.immuneml.uio.no/tutorials/how_to_specify_an_analysis_with_yaml.html).

### Command line usage 

The `immune-ml` command takes only two parameters: the YAML specification file and a result path. 
An example is given here:

```bash
immune-ml path/to/specification.yaml result/folder/path/
```

For each instruction specified in the YAML specification file, a subfolder is created in the 
`result/folder/path`. Each subfolder will contain:
- An `index.html` file which shows an overview of the results produced by that instruction. Inspecting the results of an immuneML analysis typically starts here. 
- A copy of the used YAML specification (`full_specification.yaml`) with all default parameters explicitly set.
- A folder containing all raw results produced by the instruction.
- A folder containing the imported dataset(s) in optimized binary (Pickle) format.

## Requirements
(For running Generative Models)
- [Python 3.7 or 3.8](https://www.python.org/)
- Python packages:
   - [PyYAML](https://pyyaml.org) (5.3 or higher)
   - [airr](https://pypi.org/project/airr/) (1 or higher)
   - [dill](https://pypi.org/project/dill/) (0.3 or higher)
   - [editdistance](https://pypi.org/project/editdistance/) (0.5.3 or higher)
   - [fishersapi](https://pypi.org/project/fishersapi/)
   - [gensim](https://pypi.org/project/gensim/) (3.8 or higher, < 4)
   - [h5py](https://www.h5py.org/) (2.10.0 or lower when using the optional DeepRC dependency)
   - [logomaker](https://pypi.org/project/logomaker/) (0.8 or higher)
   - [matplotlib](https://matplotlib.org) (3.1 or higher)
   - [matplotlib-venn](https://pypi.org/project/matplotlib-venn/) (0.11 or higher)
   - [numpy](https://www.numpy.org/) (1.18 or higher)
   - [pandas](https://pandas.pydata.org/) (1 or higher)
   - [plotly](https://plotly.com/python/) (4 or higher)
   - [pystache](https://pypi.org/project/pystache/) (0.5.4)
   - [Pytorch](https://pytorch.org/) (1.5.1 or higher)
   - [scikit-learn](https://scikit-learn.org/) (0.23 or higher)
   - [scipy](https://www.scipy.org)
   - [tqdm](https://tqdm.github.io/) (0.24 or higher)
   - [tzlocal](https://pypi.org/project/tzlocal/)
   - [tensorboard](https://pypi.org/project/tensorboard/) (1.14.0 or higher)
   - [Cython](https://pypi.org/project/Cython/)
   - [tensorflow](https://pypi.org/project/tensorflow/)
   - [pyprind](https://pypi.org/project/PyPrind/)
   - [keras-tuner](https://pypi.org/project/keras-tuner/)

# Citing immuneML

If you are using immuneML in any published work, please cite:

Pavlović, M., Scheffer, L., Motwani, K. et al. The immuneML ecosystem for machine learning analysis of adaptive immune 
receptor repertoires. Nat Mach Intell 3, 936–944 (2021). https://doi.org/10.1038/s42256-021-00413-z



<hr>


© Copyright 2021-2022, Milena Pavlovic, Lonneke Scheffer, Keshav Motwani, Victor Greiff, Geir Kjetil Sandve


