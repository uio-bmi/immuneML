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



# Clustering
  This branch is a developer branch for clustering in immuneML. If you don't plan to use this 
  feature use the master branch instead.

  Before using clustering in immuneML make sure immuneML is installed correctly and working by 
  following the official installation guide on https://docs.immuneml.uio.no/latest/installation.html.
  
  Additionally, if you want to use the KMedoids clustering method you have to install 
  [scikit-learn-extra](https://github.com/scikit-learn-contrib/scikit-learn-extra) (0.2 or higher).
  This can be done using PyPi with the command:

```bash 
pip install scikit-learn-extra
```
  

## Usage
### Quickstart
  Here is a quickstart guide on how to get started using clustering in immuneML. 
  For now this can only be done using the command line interface.
#### Command line usage
  To run immuneML through the command line we need to specify what yaml file
  to use as well as a path to the output folder:
```bash 
immune-ml .\clustering_quickstart.yaml .\output\
```
  The quickstart yaml contains instructions to do clustering on the EBV.tsv file. It will do 3 different analyses, 
  one with the data reduced to 2 dimensions, one with the data reduced to 3 dimensions and 
  a third one where we don't use dimensionality reduction at all.
```yaml
definitions:
  datasets:
    d1: # user-defined dataset name
      format: VDJdb # dataset format
      params:
        path: EBV.tsv # path to the folder containing the receptor .tsv file
        is_repertoire: False  # we are importing a receptor dataset
        paired: True   # data is paired TRA to TRB
        receptor_chains: TRA_TRB
  encodings:
    3mer_encoding:  # user-defined encoding name
      KmerFrequency:  # encoding type
        k: 3  # encoding parameters
  dimensionality_reduction:
    pca_2d: # user-defined dimensionality reduction name
      PCA:  # dimensionality reduction type
        n_components: 2 #number of dimensions to reduce to
    pca_3d:
      PCA:
        n_components: 3
  ml_methods:
    kmeans: # user-defined ml method name
      KMeans: # clustering method type
        n_clusters: 5 # number of wanted clusters
  reports:
    clusteringReport:  # user-defined report name
      ClusteringReport: # report type
        labels: # labels in data to compare clustering to
          - epitope
instructions:
  clustering_inst1: # user-defined instruction name
    type: Clustering  # instruction type
    analyses: # all analyses to run
      kmeans_2d:  # user-defined analysis name
        dataset: d1 # dataset to run this analysis on
        encoding: 3mer_encoding # encoding to use for this analysis
        dimensionality_reduction: pca_2d # dimensionality reduction to run before clustering (optional)
        clustering_method: kmeans # clustering method to use for this analysis
        report: clusteringReport # report name for this analysis
      kmeans_3d:
        dataset: d1
        encoding: 3mer_encoding
        dimensionality_reduction: pca_3d
        clustering_method: kmeans
        report: clusteringReport
      kmeans_no_dim_reduction:
        dataset: d1
        encoding: 3mer_encoding
        clustering_method: kmeans
        report: clusteringReport
    number_of_processes: 4  # processes for parallelization
```
  The example data the yaml refers to (EBV.tsv) as well as the yaml file can already be found on the branch.

  For a detailed explanation of the YAML specification file, see the tutorial [How to specify an analysis with YAML](https://docs.immuneml.uio.no/tutorials/how_to_specify_an_analysis_with_yaml.html).

### Expected result
  In the output folder you will now have a file named index.html. This file will open the report 
  created by the command ran above.
  What you will see here is first a summary of the three analyses and what analyses scored best when looking at 
  the [Silhouette Coefficient](https://en.wikipedia.org/wiki/Silhouette_(clustering)), 
  [Calinski-Harabasz index](https://medium.com/@haataa/how-to-measure-clustering-performances-when-there-are-no-ground-truth-db027e9a871c#:~:text=complexity%3A%20O(n%C2%B2)-,Calinski%2DHarabasz%20Index,-The%20Calinski%2DHarabasz) and
  [Davies-Bouldin index](https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index).

  After that all the analyses are presented. Here you have a some dropdowns with details of the dataset, encoding, 
  clustering algorithm parameters, dimensionality reduction parameters (if used) and 
  the scores talked about earlier for this particular analysis.
  
  If the data has been reduced to either two or three dimensions you will have a scatter plot next
  showing all the different data points color coded for what cluster they are in. You can hover over a point to 
  see what cluster they represent.
  The next graph is a heatmap showing the comparison between the label specified in the yaml(epitope) 
  and the cluster id they received. This is a representation of how many percent of the different 
  available values for the specified label are placed in what cluster. This can be used to see if the data
  has clustered based on something like epitope, MHC, or any other attribute that is represented in the data.
  
  Finally, the dataset can be downloaded in AIRR format with the cluster id added.
  
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
- Optional dependencies when using KMediods Clustering:
   - [scikit-learn-extra](https://github.com/scikit-learn-contrib/scikit-learn-extra) (0.2.0 or higher)

# Citing immuneML

If you are using immuneML in any published work, please cite:

Pavlović, M., Scheffer, L., Motwani, K. et al. The immuneML ecosystem for machine learning analysis of adaptive immune 
receptor repertoires. Nat Mach Intell 3, 936–944 (2021). https://doi.org/10.1038/s42256-021-00413-z



<hr>


© Copyright 2021-2022, Milena Pavlovic, Lonneke Scheffer, Keshav Motwani, Victor Greiff, Geir Kjetil Sandve


