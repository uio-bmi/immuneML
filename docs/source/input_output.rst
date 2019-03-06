#######################
Import and export data
#######################

.. toctree::
   :maxdepth: 2

This section describes how to import the data into ImmuneML and how to export it.

Import
======

The data can be imported from the following formats:

1. MiXCR_

MiXCR
-----

To load the data that is available as an output from MiXCR, use **MiXCRLoader** class. An example is given below.

.. code-block:: python

    dataset = MiXCRLoader.load(path="./input_files/", params={
        "additional_columns": ["minQualCDR3"],
        "sequence_type": "CDR3",
        "result_path": "./output_files/",
        "batch_size": 2,
        "extension": "csv",
        "custom_params": [{
            "name": "CD",
            "location": "filepath_binary",
            "alternative": "HC"
        }]
    })

The ``load()`` method takes in the path to the root folder where MiXCR files are stored and a dictionary with custom
parameters as the second argument.

The ``path`` parameter should point to the root folder of MiXCR files, since the ``load()`` method will recursively
discover all files in that path having the extension specified in the ``params``.

The ``params`` dictionary defines how to perform the import. It consists of the following fields:

1.  ``additional_columns``: by default, only V gene, J gene, clone count, patient, sample ID and the nucleotide and
amino acid sequences of the given sequence type are stored. Other fields are stored in **SequenceMetadata** object only
if they are specifically mentioned by their MiXCR name in ``additional_columns``. ``additional_columns`` is a list of
field names that will be stored for the sequence.

2.  ``sequence_type``: defines which immune receptor region should be stored in the receptor sequence objects to be used
for the subsequent analysis. In this example, it is ``CDR3``, but can be any other region name which is provided in the
MiXCR output and which is not ``NA`` for all the sequences.

3.  ``result_path``: this is the path on which the loaded repertoires and the dataset will be stored.

4.  ``batch_size``: defines how many repertoires can be loaded at once and is a trade-off between the repertoire size and
the speed of input/output operations. In general, the larger the repertoires, the smaller this value should be. The smallest
possible value is 1, meaning that only one repertoire at the time will be loaded.

5.  ``extension``: defines the type of file ``load()`` will be looking for when discovering MiXCR files on the given ``path``.
This is typically ``csv``, but can be any other type as well.

6.  ``custom_params``: this is a list of parameters on the repertoire level that are not directly present in the MiXCR
output files, but are necessary for the analysis. For example, if two groups of repertoires are put into two different folders
each of which contains a disease status in its name (e.g. CD_files/ for repertoires with celiac disease and HC_files/ for
repertoires of healthy controls), then this defines what the ``name`` of that parameter is, the ``location`` it can be found at and what the
``alternative`` value is if this is a binary case. This option with writing the disease status explicitly in the subfolder name
is the only one supported so far for ``custom_params``.

The ``load()`` function will return a valid dataset object which is also stored as a pickle file in the given ``result_path``.

.. _MiXCR: https://mixcr.readthedocs.io/en/master/index.html
