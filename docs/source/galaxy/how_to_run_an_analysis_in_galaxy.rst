How to run an analysis in Galaxy
================================

To run an analysis in Galaxy, ImmuneML Wrapper tool should be used. This tool can run
any analysis supported by immuneML. For the tool, it is necessary to provide a YAML
specification file defining the analysis, and a list of input files (if any) or a
list of Galaxy collections (if any).

YAML specification describes the task which should be executed. To see the details on how
to write the specification, see :ref:`How to specify an analysis with YAML`. The differences from the specification when running
immuneML as a command line tool and running it in Galaxy are:

1. The paths to files should only include filenames as all files provided will be
available in the working directory.

2. If using a Galaxy collection (which was created by immuneML dataset tool) as a dataset,
in the YAML it should be specified that it is in Pickle format and only the name of the
metadata file should be provided under params field. The metadata file name can be seen
in the collection.

To use additional files or Galaxy collections, these files need to be present in the current Galaxy history.
