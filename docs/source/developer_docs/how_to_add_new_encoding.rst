
How to add a new encoding
===========================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML dev docs: add a new encoding
   :twitter:description: See how to add a new encoding to the immuneML platform.
   :twitter:image: https://docs.immuneml.uio.no/_images/extending_immuneML.png



.. include:: ./coding_conventions_and_tips.rst



Adding a new encoder class
--------------------------

For this example, we provide a :code:`SillyEncoder` <todo more text>

.. collapse:: SillyEncoder.py

  .. code:: python

     print("to be added")


#. Add a new `Python package <https://packaging.python.org/en/latest/tutorials/packaging-projects/>`_ to the :py:obj:`~immuneML.encodings` package.

#. Add a new encoder class to the package. The new class should inherit from the base class :py:obj:`~immuneML.encodings.DatasetEncoder.DatasetEncoder`.
   The name of the class should end with 'Encoder', and when calling this class in the YAML specification, the 'Encoder' suffix is omitted.
   In the test example below, the class is called :code:`SillyEncoder`, which would be referred to as :code:`Silly` in the YAML specification.

#. Implement all abstract methods from the :py:obj:`~immuneML.encodings.DatasetEncoder.DatasetEncoder` class.
   See the class documentation for more detailed descriptions of the implementation of these methods.
   See also :ref:`Implementing the encode() method in a new encoder class`.
   In the :code:`SillyEncoder` example code, these have already been implemented.

#. Optionally: add a default parameters YAML file. This file should be added to the folder :code:`config/default_params/encodings`.
   The default parameters file is automatically discovered based on the name of the class using : the base name (without 'Encoder' suffix) converted to snake case, and with an added '_params.yaml' suffix.
   For the :code:`SillyEncoder`, this is :code:`silly_params.yaml`, which could for example contain the following:

   .. code:: yaml

      random_seed: 1

   In rare cases where classes have unconventional names that do not translate well to CamelCase (e.g., MiXCR, VDJdb), this needs to be accounted for in :py:meth:`~immuneML.dsl.DefaultParamsLoader.convert_to_snake_case`.

#. If a compatible ML method is already available, add the new encoder class to the list of compatible encoders returned by the
   :code:`get_compatible_encoders()` method of the :py:obj:`~immuneML.ml_methods.MLMethod.MLMethod` of interest.
   See also :ref:`Adding encoder compatibility to an ML method`.

#. Add class documentation, see :ref:`Class documentation standards`.

#. Add unit tests, see :ref:`Adding a unit test for the new encoder`.

#. Test run the encoder. A suggestion for a minimal YAML example is given below.
   This example analysis creates a randomly generated dataset, encodes the data using the :code:`SillyEncoder`
   and exports the encoded data as a csv file.

   .. collapse:: test_run_silly_encoder.yaml

      .. code:: yaml

         definitions:
           datasets:
             my_dataset:
               format: RandomSequenceDataset
               params:
                 sequence_count: 100
                 labels:
                   binds_epitope:
                     True: 0.6
                     False: 0.4

           encodings:
             my_silly_encoder:
               Silly:
                 random_seed: 3

           reports:
             my_design_matrix: DesignMatrixExporter

         instructions:
           my_instruction:
             type: ExploratoryAnalysis
             analyses:
               my_analysis_1:
                 dataset: my_dataset
                 encoding: my_silly_encoder
                 report: my_design_matrix
                 labels:
                 - binds_epitope


Encoders for different dataset types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inside immuneML, three different types of datasets are considered: :py:obj:`~immuneML.data_model.dataset.RepertoireDataset.RepertoireDataset` for immune
repertoires, :py:obj:`~immuneML.data_model.dataset.SequenceDataset.SequenceDataset` for single-chain immune
receptor sequences and :py:obj:`~immuneML.data_model.dataset.ReceptorDataset.ReceptorDataset` for paired sequences.
Encoding should be implemented separately for each dataset type. This can be solved in two different ways:

- Have a single Encoder class containing separate methods for encoding different dataset types.
  During encoding, the dataset type is checked, and the corresponding methods are called.
  An example of this is given in the SillyEncoder :ref:`Adding a new encoder class`.

- Have an abstract base Encoder class for the general encoding type, with subclasses for each dataset type.
  The base Encoder contains all shared functionalities, and the subclasses contain dataset-specific functionalities,
  such as code for 'encoding an example'.
  Note that in this case, the base Encoder implements the method :code:`build_object(dataset: Dataset, params)`,
  that returns the correct dataset type-specific encoder subclass.
  An example of this is :py:obj:`~immuneML.encodings.onehot.OneHotEncoder.OneHotEncoder`.


When an encoding only makes sense for one possible dataset type, only one class needs to be created.
The :code:`build_object(dataset: Dataset, params)` method should raise a user-friendly error when an illegal dataset type is supplied.
An example of this can be found in :py:obj:`~immuneML.encodings.motif_encoding.SimilarToPositiveSequenceEncoder.SimilarToPositiveSequenceEncoder`.


Implementing the encode() method in a new encoder class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The encode() method is called by immuneML to encode a new dataset. This method should be called with two arguments: a dataset and params
(an :py:obj:`~immuneML.encodings.EncoderParams.EncoderParams` object). The EncoderParams objects contains useful information such as the path where optional files with intermediate data can be
stored (such as vectorizer files, normalization, etc.), a :py:obj:`~immuneML.environment.LabelConfiguration.LabelConfiguration` object containing the
labels that were specified for the analysis, and more.

The :code:`encode()` method should return a new dataset object, which is a copy of the original input dataset, but with an added :code:`encoded_data` attribute.
The :code:`encoded_data` attribute should contain an :py:obj:`~immuneML.data_model.encoded_data.EncodedData.EncodedData` object, which is created with the
following arguments:

  - examples: a design matrix where the rows represent Repertoires, Receptors or Sequences ('examples'), and the columns the encoding-specific features.
  - encoding: a string denoting the encoder base class that was used.
  - labels: a dictionary of labels, where each label is a key, and the values are the label values across the examples (for example: {disease1: [positive, positive, negative]} if there are 3 repertoires). This parameter should be set only if :code:`EncoderParams.encode_labels` is True, otherwise it should be set to None.
  - example_ids: a list of identifiers for the examples (Repertoires, Receptors or Sequences). This can be retrieved using :code:`Dataset.get_example_ids()`.
  - feature_names: a list of feature names, i.e., the names given to the encoding-specific features. When included, list must be as long as the number of features.
  - feature_annotations: an optional pandas dataframe with additional information about the features. When included, number of rows in this dataframe must correspond to the number of features. This parameter is not typically used.
  - info: an optional dictionary that may be used to store any additional information that is relevant (for example paths to additional output files). This parameter is not typically used.

The :code:`examples` attribute of the :code:`EncodedData` objects will be directly passed to the ML models for training. Other attributes are used for reports and
interpretability.


<to add: special cases (extra files, full dataset)>



Adding a unit test for the new encoder
----------------------------------------

to be added


Class documentation standards
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ./class_documentation_standards.rst


