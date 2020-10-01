How to make an immuneML dataset in Galaxy
=========================================

immuneML dataset tool allows users to create an immuneML dataset in Galaxy.
From immune repertoire files in MiXCR, Adaptive Biotechnologies or AIRR format or
receptor files as extracted from VDJdb, a Galaxy collection (a list of immuneML-imported
files) is created and can be used as input for the immuneML wrapper tool.

The tool has three input fields:

1. YAML specification,

2. Metadata file (optional)

3. A list of files to import to a dataset.

YAML specification defines how the dataset should be created from supplied files. See :ref:`Specification` for more details on writing a YAML
specification file, specifically with :ref:`DatasetGeneration` instruction.. For this Galaxy tool, the specification will include only one dataset
and only one format in which it will be exported.

Metadata file field is used when creating a dataset consisting of immune repertoires
and describes the metadata information for one repertoire per row. For the format of
the metadata file, see :ref:`What should the metadata file look like?`. For datasets consisting of immune receptor or receptor
sequences, the metadata information is provided directly in the files including receptors
as separate fields (e.g. epitope field in VDJdb format).

The third field includes all data files. For repertoire datasets, a repertoire file
corresponding to each metadata row will be imported (if a repertoire was specified in
the metadata but the corresponding repertoire file is not present, the tool will raise
an error). For receptor or receptor sequence datasets, metadata file is not used, but
instead all data from the corresponding format is imported along with its metadata.
