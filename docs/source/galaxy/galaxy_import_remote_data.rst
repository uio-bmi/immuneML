How to import remote AIRR datasets in Galaxy
===========================================================
The immuneML Galaxy web interface integrates with the iReceptor Gateway and VDJdb such that data can automatically be sent to the Galaxy history,
and subsequently an immuneML dataset can be generated through the `Create dataset <https://galaxy.immuneml.uio.no/root?tool_id=immune_ml_dataset>`_ tool.

We strongly recommend to:

- use the iReceptor Gateway to retrieve repertoire datasets, which are annotated with metadata per repertoire, for example disease status
- use VDJdb to retrieve sequence or receptor datasets, which are annotated with labels per receptor sequence, such as binding epitope


How to import data from the iReceptor Gateway into an immuneML RepertoireDataset
--------------------------------------------------------------------------------

See `this example Galaxy history <https://galaxy.immuneml.uio.no/u/immuneml/h/create-ireceptor-dataset>`_ showing how to make a dataset from the iReceptor Gateway into an immuneML dataset.

Retrieving the dataset
^^^^^^^^^^^^^^^^^^^^^^
Under 'Get Remote Data', use the iReceptor Plus Gateway tool to search for and download the desired dataset.
Note that you must apply for a user account to be able to use the iReceptor Gateway.
See `the iReceptor Gateway documentation <http://ireceptor.irmacs.sfu.ca/platform/doc>`_ for more details.
On the Downloads page on the iReceptor Gateway website, you should find your downloaded dataset and a button named 'Send to Galaxy →'
A new .zip file should now appear in the Galaxy history.
By clicking 'View data' (eyeball icon), the .zip file can be downloaded and inspected locally if desired. The .zip should
contain one info.txt file, one or more .tsv files, and for each .tsv file there should be a corresponding file with -metadata.json extension.


Combining multiple datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^
It is possible to collect multiple iReceptor datasets into one immuneML dataset, for example when combining one dataset
with repertoires from diseased subjects, and a second dataset from healthy subjects. In this case, make sure that each of the
.zip files must have a unique name in the history, as duplicate names will be ignored by the `Create dataset <https://galaxy.immuneml.uio.no/root?tool_id=immune_ml_dataset>`_ tool.
The names of  history elements can be altered by clicking their 'Edit attributes' button (pencil icon).


Creating the immuneML dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Under immuneML, use the `Create dataset <https://galaxy.immuneml.uio.no/root?tool_id=immune_ml_dataset>`_ tool to create the immuneML dataset (see :ref:`How to make an immuneML dataset in Galaxy` for a more general explanation of this tool).
Select the iReceptor Gateway data format, and select one or more .zip files from the Galaxy history.
A difference between creating an immuneML repertoire dataset from iReceptor Gateway .zip files and other file formats is that in the former case, no metadata file needs to be provided.
The metadata file will internally be constructed by immuneML based on the metadata.json files provided in the .zip file.
It is therefore important to manually inspect the generated metadata file for an immuneML dataset, to verify that the metadata has been inferred as expected, and to understand
which labels are available for Repertoire classification. This can be done by downloading the generated dataset and inspecting the dataset_metadata.csv file.
See :ref:`What should the metadata file look like?` to read more about the formatting of the metadata file.


Why does the number of repertoires in my immuneML dataset exceed the number of .tsv files in the iReceptor Gateway .zip?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The AIRR .tsv files contained in .zip provided by the iReceptor Gateway may contain data from several different immune repertoires (different donors) simultaneously.
For immuneML, these repertoires must each be in their own individual file, and the airr .tsv file may therefore be split into multiple different files upon import.
Furthermore, according to `the AIRR schema <https://docs.airr-community.org/en/stable/datarep/metadata.html>`_, a Repertoire may be associated with multiple different samples and data processings.
During iReceptor data import, the data is split such that each repertoire is associated with one repertoire_id, one sample_processing_id, and one data_processing_id.
This information can be found in the generated dataset_metadata.csv file.

If you want to use an alternative way of defining which repertoires to use for the immuneML dataset (e.g., only including data from one data_processing_id, or combining data
from multiple sample_processing_id's together into repertoires), this can be done by:

- Downloading the iReceptor Gateway .zip file and manually splitting the repertoires and creating the metadata file, and then importing the data in AIRR format.
- Exporting the immuneML dataset to AIRR format (see :ref:`Using the advanced 'Create dataset' interface` and use AIRR export format), downloading and editing this dataset, and then importing the data in AIRR format.



How to import data from VDJdb into an immuneML Receptor- or SequenceDataset
---------------------------------------------------------------------------

See `this example Galaxy history <https://galaxy.immuneml.uio.no/u/immuneml/h/create-vdjdb-dataset>`_ showing how to retrieve data from VDJdb and turn it into an immuneML sequence or receptor dataset.


Retrieving the dataset
^^^^^^^^^^^^^^^^^^^^^^
Under 'Get Remote Data', use the Retrieve data from VDJdb tool to search for the desired dataset.
This interface mimics `the original VDJdb browser <https://vdjdb.cdr3.net/search>`_, and provides the same options.
For receptor datasets, make sure to set 'Append pairs' to 'yes'. Only complete TCRs (where both alpha and beta chains were retrieved) can
be imported as receptors.
The imported file can be inspected by clicking 'View data' (eyeball icon).


Combining multiple datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^
It is possible to collect multiple VDJdb files into one immuneML dataset, for example when combining one file with sequences binding
to a given epitope, and a second file with sequences that do not bind. In this case, make sure that each of the files has a unique
name in the history, as duplicate names will be ignored by the `Create dataset <https://galaxy.immuneml.uio.no/root?tool_id=immune_ml_dataset>`_ tool.
The names of  history elements can be altered by clicking their 'Edit attributes' button (pencil icon).


Creating the immuneML dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Under immuneML, use the `Create dataset <https://galaxy.immuneml.uio.no/root?tool_id=immune_ml_dataset>`_ tool to create the immuneML dataset (see :ref:`How to make an immuneML dataset in Galaxy` for a more general explanation of this tool), select
one or more files from the history and use the VDJdb data format.
VDJdb files contain the columns Epitope, Epitope gene and Epitope species which may be filled in under the field Metadata columns such that these fields can be used as receptor sequence classification labels.
