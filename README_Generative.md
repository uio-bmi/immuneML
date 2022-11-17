# immuneML

![Python application](https://github.com/uio-bmi/immuneML/workflows/Python%20application/badge.svg?branch=master)
![Docker](https://github.com/uio-bmi/immuneML/workflows/Docker/badge.svg?branch=master)
[![](https://img.shields.io/static/v1?label=AIRR-C%20sw-tools%20v1&message=compliant&color=008AFF&labelColor=000000&style=plastic)](https://docs.airr-community.org/en/stable/swtools/airr_swtools_standard.html)


immuneML is a platform for machine learning-based analysis and 
classification of adaptive immune receptors and repertoires (AIRR).

This branch of immuneML focuses on Generative Models and their use within immunology.
Following is an introduciton to the Generative Models that have been included and how 
to utilize them. As this is a WIP this document will continually be updated as it is developed.
If any information in this document is out of date please inform us as soon as possible.

In order to utilize the generative models download this branch of immuneML, either downloading as a zip, or cloning the 
repository.

```bash
git clone https://github.com/uio-bmi/immuneML
```

Then checkout to the correct branch (name is also WIP):

```bash
git checkout GenerativeModelsWUnsupervisedLearning
```

## Usage 

immuneML as a whole has many unique usages, most of which is not touched upon within this documentation. However, if 
you wish to know more about these see the README.md in the master branch: https://github.com/uio-bmi/immuneML

There are two main usages of Generative Models in immuneML, training and generation.
Both generates as many sequences as specified, or 10 if not specified.
The difference in the two is one requires a dataset in order to train the model first, while the other can take a finished
model as input.

### Quickstart

Within the Generative Models branch are a few files and directories, not found elsewhere in immuneML which can be
used to run a quickstart.<br/>
There are two YAML specifications, one for training and one for generation, and two directories, one relevant for each 
YAML.

The file generative_quickstart.yaml runs the training of a PWM model and generates some sequences displayed in a report
This file uses the data found in generative_model_data to train. This specification relays the minimum needed to run
the generative models. An optional parameter for both instrucitons is the "amount" parameter, which determines the number
of sequences to produce.

#### generative_quickstart.yml
```yaml
definitions:
  datasets:
    d1:
      format: AIRR
      params:
        path: generative_model_data\my_dataset_export_instruction\d1\AIRR
        metadata_file: generative_model_data\my_dataset_export_instruction\d1\AIRR\metadata.csv
        is_repertoire: True
  ml_methods:
    G1:
      PWM:
        cores_for_training: 2
  reports:
    GeneratorReport: GeneratorReport
instructions:
  machine_learning_instruction:
    type: GenerativeModel
    generators:
      generator_1:
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
      PWM:
        cores_for_training: 2
  reports:
    GeneratorReport: GeneratorReportPWM
instructions:
  machine_learning_instruction:
    type: GenerativeModelLoad
    generators:
      generator_1:
        ml_method: G1
        report: GeneratorReport
        path: existing_PWM_model\machine_learning_instruction\analysis_generator_1
```

#### Report

So far there are two different reports that can be used on the genrative models, GeneratorReport and GeneratorReportPWM.
GeneratorReport can be used on all generative models but relays little information. While GeneratorReportPWM is tailored 
for the PWM and displayes useful graphics. These reports can be found in the output directory specified when running the
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

# Citing immuneML

If you are using immuneML in any published work, please cite:

Pavlović, M., Scheffer, L., Motwani, K. et al. The immuneML ecosystem for machine learning analysis of adaptive immune 
receptor repertoires. Nat Mach Intell 3, 936–944 (2021). https://doi.org/10.1038/s42256-021-00413-z



<hr>


© Copyright 2021-2022, Milena Pavlovic, Lonneke Scheffer, Keshav Motwani, Victor Greiff, Geir Kjetil Sandve


