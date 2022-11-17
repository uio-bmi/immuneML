# immuneML

![Python application](https://github.com/uio-bmi/immuneML/workflows/Python%20application/badge.svg?branch=master)
![Docker](https://github.com/uio-bmi/immuneML/workflows/Docker/badge.svg?branch=master)
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

# Dimensionality reducion

This is a branch dedicated to **dimensionality reduction**, so if you are looking for installation guide or other general information refer to the readme on the Main branch: https://github.com/uio-bmi/immuneML

## Usage 

### Quickstart

Here we will go over how to a quickstart run in order to get started using dimensionality reduction in immuneML

#### Command line usage 

To run immuneML thorugh the command line we need to specify what yaml file to use as well as a path to the output folder:

```bash
immune-ml .\dim_reduction_quickstart.yaml .\output\
```

The quickstart yaml:

```yaml
definitions:
  datasets:
    d1:
      format: VDJdb # specify what format the data is in
      params:
        path: AVF.tsv # path to dataset
        is_repertoire: False # no, this is receptor dataset
        paired: True
        receptor_chains: TRA_TRB # what chain pair to import for a ReceptorDataset
  encodings:
    kmer_encoding: KmerFrequency
  reports: # Specifying what report to use
    dim_red: # user-defined report name
      DimensionalityReduction # Which report to use
  dimensionality_reduction: # Specifying what dimensionality reduction method(s) to use
    pca: # user-defined instruction name
      PCA: # Dimensionality reduction method name
        n_components: 2 # user-defined parameters
instructions:
  my_expl_analysis_instruction: # user-defined instruction name
    type: ExploratoryAnalysis # which instruction to execute
    analyses: # analyses to perform
      my_analysis: # user-defined name of the analysisy
        dataset: d1 # dataset to use in the first analysis
        encoding: kmer_encoding # what encoding to use on the dataset
        dimensionality_reduction: pca # what dimensionality reduction method to use
        report: dim_red # which report to generate using the dataset d1
    number_of_processes: 4
```
The example data the yaml data refers to (AVF.tsv) as well as the yaml file can already be found on the branch.

For a detailed explanation of the YAML specification file, see the tutorial [How to specify an analysis with YAML](https://docs.immuneml.uio.no/tutorials/how_to_specify_an_analysis_with_yaml.html).

#### Expected result
When opening the html file the expected result should be something like this:
![quickstart_expected_result_pca_plot](https://user-images.githubusercontent.com/33656177/202473145-f8c5eb10-6750-4fe2-8f7f-9448d8bf3f15.PNG)
![quickstart_expected_result_pca_explained](https://user-images.githubusercontent.com/33656177/202472563-3f606020-15c0-49f4-a146-6e933eb4975d.PNG)

## Support

We will prioritize fixing important bugs, and try to answer any questions as soon as possible. We may implement suggested features and enhancements as time permits. 

If you run into problems when using immuneML, please see [the documentation](https://docs.immuneml.uio.no/latest/). In particular, we recommend you check out:
- The [Quickstart tutorial](https://docs.immuneml.uio.no/latest/quickstart.html) for new users
- The [Troubleshooting](https://docs.immuneml.uio.no/latest/troubleshooting.html) page

If this does not answer your question, you can contact us via:
- Twitter [`@immuneml`](https://twitter.com/immuneml)
- Email [`contact@immuneml.uio.no`](mailto:contact@immuneml.uio.no)

To report a potential bug or suggest new features, please [submit an issue on GitHub](https://github.com/uio-bmi/immuneML/issues).

If you would like to make contributions, for example by adding a new ML method, encoding, report or preprocessing, please [see our developer documentation](https://docs.immuneml.uio.no/latest/developer_docs.html) and [submit a pull request](https://github.com/uio-bmi/compairr/pulls).

## Requirements

- [Python 3.7 or 3.8](https://www.python.org/)
- Python packages:
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
   - [PyYAML](https://pyyaml.org) (5.3 or higher)
   - [regex](https://pypi.org/project/regex/) 
   - [requests](https://requests.readthedocs.io/) (2.21 or higher)
   - [scikit-learn](https://scikit-learn.org/) (0.23 or higher)
   - [scipy](https://www.scipy.org)
   - [tensorboard](https://www.tensorflow.org/tensorboard) (1.14.0 or higher)
   - [tqdm](https://tqdm.github.io/) (0.24 or higher)
   - [tzlocal](https://pypi.org/project/tzlocal/) 
- Optional dependencies when using DeepRC:
   - [DeepRC](https://github.com/ml-jku/DeepRC) (0.0.1)
   - [widis-lstm-tools](https://github.com/widmi/widis-lstm-tools) (0.4)
- Optional dependencies when using TCRdist:
   - [parasail](https://pypi.org/project/parasail/) (1.2)
   - [tcrdist3](https://github.com/kmayerb/tcrdist3) (0.1.6 or higher)

# Citing immuneML

If you are using immuneML in any published work, please cite:

Pavlović, M., Scheffer, L., Motwani, K. et al. The immuneML ecosystem for machine learning analysis of adaptive immune 
receptor repertoires. Nat Mach Intell 3, 936–944 (2021). https://doi.org/10.1038/s42256-021-00413-z



<hr>


© Copyright 2021-2022, Milena Pavlovic, Lonneke Scheffer, Keshav Motwani, Victor Greiff, Geir Kjetil Sandve


