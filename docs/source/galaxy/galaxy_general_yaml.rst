How to run any AIRR ML analysis in Galaxy
=========================================

To run an analysis in Galaxy, `Run immuneML with YAML specification` tool should be used. This tool can run
any analysis supported by immuneML. To use the tool, it is necessary to provide a YAML
specification file defining the analysis, and a list of input files (if any) or a
list of Galaxy collections (if any).

YAML specification describes the task which should be executed. To see the details on how
to write the specification, see :ref:`How to specify an analysis with YAML`. The differences from the specification when running
immuneML as a command line tool and running it in Galaxy are:

- When running from Galaxy, the paths to files should only include filenames as all files provided will be available in the working directory.

- When using a Galaxy collection (as created by the `Create dataset` tool), as a dataset, the metadata filename can be found in the
  collection, along with other files. This depends on the format the dataset was created in. See :ref:`Datasets` for more information
  on how to specify import parameters of a dataset in different formats (currently, dataset stored as a Galaxy collection will be in
  either Pickle or AIRR format).

To use additional files or Galaxy collections, these files need to be present in the current Galaxy history.
