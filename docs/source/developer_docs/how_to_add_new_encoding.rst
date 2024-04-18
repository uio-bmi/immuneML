
How to add a new encoding
===========================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML dev docs: add a new encoding
   :twitter:description: See how to add a new encoding to the immuneML platform.
   :twitter:image: https://docs.immuneml.uio.no/_images/extending_immuneML.png


.. include:: ./coding_conventions_and_tips.rst


Adding a new DatasetEncoder class
------------------------------------

For a detailed documentation of how to implement each method, as well as some special cases, please see the docstrings of the :py:obj:`~immuneML.encodings.DatasetEncoder.DatasetEncoder` class.

Adding a first example encoder
******************************

Example: SillyEncoder
^^^^^^^^^^^^^^^^^^^^^^^^

For this tutorial, we provide a :code:`SillyEncoder` (:download:`download here <./example_code/SillyEncoder.py>` or view below), in order to test adding a new Encoder file to immuneML.
This encoder ignores the data of the input examples, and generates a few random features per example.

.. collapse:: View SillyEncoder.py

  .. literalinclude:: ./example_code/SillyEncoder.py
     :language: python



Step-by-step tutorial
^^^^^^^^^^^^^^^^^^^^^^^^


#. Add a new `Python package <https://packaging.python.org/en/latest/tutorials/packaging-projects/>`_ to the :py:obj:`~immuneML.encodings` package.
   This means: a new folder (with meaningful name) containing an empty :code:`__init__.py` file.

#. Add a new encoder class to the package. The new class should inherit from the base class :py:obj:`~immuneML.encodings.DatasetEncoder.DatasetEncoder`.
   The name of the class should end with 'Encoder', and when calling this class in the YAML specification, the 'Encoder' suffix is omitted.
   In the test example below, the class is called :code:`SillyEncoder`, which would be referred to as :code:`Silly` in the YAML specification.

#. Implement all abstract methods from the :py:obj:`~immuneML.encodings.DatasetEncoder.DatasetEncoder` class.
   See the class documentation for more detailed descriptions of the implementation of these methods.
   See also :ref:`Implementing the encode() method in a new encoder class`.
   In the :code:`SillyEncoder` example code, these have already been implemented.

#. If the encoder has any default parameters, they should be added in a default parameters YAML file. This file should be added to the folder :code:`config/default_params/encodings`.
   The default parameters file is automatically discovered based on the name of the class using the base name (without 'Encoder' suffix) converted to snake case, and with an added '_params.yaml' suffix.
   For the :code:`SillyEncoder`, this is :code:`silly_params.yaml`, which could for example contain the following:

   .. code:: yaml

      random_seed: 1
      embedding_len: 5

   In rare cases where classes have unconventional names that do not translate well to CamelCase (e.g., MiXCR, VDJdb), this needs to be accounted for in :py:meth:`~immuneML.dsl.DefaultParamsLoader.convert_to_snake_case`.

#. Use the automated script `check_new_encoder.py <https://github.com/uio-bmi/immuneML/blob/master/scripts/check_new_encoder.py>`_ to test
   the newly added encoder. This script will throw errors or warnings if the DatasetEncoder class implementation is incorrect.
   Example command to test the :code:`SillyEncoder` for sequence datasets:

   .. code:: bash

      python3 ./scripts/check_new_encoder.py -e ./immuneML/encodings/silly/SillyEncoder.py -d sequence

#. If a compatible ML method is already available, add the new encoder class to the list of compatible encoders returned by the
   :code:`get_compatible_encoders()` method of the :py:obj:`~immuneML.ml_methods.MLMethod.MLMethod` of interest.
   See also :ref:`Adding encoder compatibility to an ML method`.

#. Finish the encoder by adding :ref:`class documentation <Encoding class documentation standards>` and :ref:`unit tests<Adding a unit test for the new encoder>`.

#. Optional: If you want to use immuneML directly to test run your encoder, the YAML example below may be used.
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


Additional details
*************************************************************

Encoders for different dataset types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



Inside immuneML, three different types of datasets are considered: :py:obj:`~immuneML.data_model.dataset.RepertoireDataset.RepertoireDataset` for immune
repertoires, :py:obj:`~immuneML.data_model.dataset.SequenceDataset.SequenceDataset` for single-chain immune
receptor sequences and :py:obj:`~immuneML.data_model.dataset.ReceptorDataset.ReceptorDataset` for paired sequences.
Encoding should be implemented separately for each dataset type. This can be solved in two different ways:

- Have a single Encoder class containing separate methods for encoding different dataset types.
  During encoding, the dataset type is checked, and the corresponding methods are called.
  An example of this is given in the SillyEncoder :ref:`Example Encoder and automatic testing`.

- Have an abstract base Encoder class for the general encoding type, with subclasses for each dataset type.
  The base Encoder contains all shared functionalities, and the subclasses contain dataset-specific functionalities,
  such as code for 'encoding an example'.
  Note that in this case, the base Encoder implements the method :code:`build_object(dataset: Dataset, params)`,
  that returns the correct dataset type-specific encoder subclass.
  An example of this is :py:obj:`~immuneML.encodings.onehot.OneHotEncoder.OneHotEncoder`, which has subclasses :py:obj:`~immuneML.encodings.onehot.OneHotSequenceEncoder.OneHotSequenceEncoder`,
  :py:obj:`~immuneML.encodings.onehot.OneHotReceptorEncoder.OneHotReceptorEncoder` and :py:obj:`~immuneML.encodings.onehot.OneHotRepertoireEncoder.OneHotRepertoireEncoder`


When an encoding only makes sense for one possible dataset type, only one class needs to be created.
The :code:`build_object(dataset: Dataset, params)` method should raise a user-friendly error when an illegal dataset type is supplied.
An example of this can be found in :py:obj:`~immuneML.encodings.motif_encoding.SimilarToPositiveSequenceEncoder.SimilarToPositiveSequenceEncoder`.


Input and output of the encode() method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The encode() method is called by immuneML to encode a new dataset.
This method is called with two arguments: a dataset and params (an :py:obj:`~immuneML.encodings.EncoderParams.EncoderParams` object), which contains:

  **EncoderParams:**

  - :code:`label_config`: a :py:obj:`~immuneML.environment.LabelConfiguration.LabelConfiguration` object containing the labels that were specified for the analysis. Should be used as an input parameter for :code:`EncoderHelper.encode_dataset_labels()`.
  - :code:`encode_labels`: boolean value which specifies whether labels must be used when encoding. Should be used as an input parameter for :code:`EncoderHelper.encode_dataset_labels()`.
  - :code:`pool_size`: the number of parallel processes that the Encoder is allowed to use, for example when using parallelisation using the package :code:`pool`. This only needs to be used when implementing parallelisation.
  - :code:`result_path`: this path can optionally be used to store intermediate files, if necessary. For most encoders, this is not necessary.
  - :code:`learn_model`: a boolean value indicating whether the encoder is called during 'training' (learn_model=True) or 'application' (learn_model=False). Thus, this parameter can be used to prevent 'leakage' of information from the test to training set. This must be taken into account when performing operations over the whole dataset, such as normalising/scaling features (example: :py:obj:`~immuneML.encodings.word2vec.Word2VecEncoder.Word2VecEncoder`). For encoders where the encoding of a single example is not dependent on other examples, (e.g., :py:obj:`~immuneML.encodings.onehot.OneHotEncoder.OneHotEncoder`), this parameter can be ignored.
  - :code:`model`: this parameter is used by e.g., :py:obj:`~immuneML.encodings.kmer_frequency.KmerFrequencyEncoder.KmerFrequencyEncoder` to pass its parameters to other classes. This parameter can usually be ignored.


The :code:`encode()` method should return a new dataset object, which is a copy of the original input dataset, but with an added :code:`encoded_data` attribute.
The :code:`encoded_data` attribute should contain an :py:obj:`~immuneML.data_model.encoded_data.EncodedData.EncodedData` object, which is created with the
following arguments:

.. include:: ./encoded_data_object.rst

The :code:`examples` attribute of the :code:`EncodedData` objects will be directly passed to the ML models for training.
Other attributes are used for reports and interpretability.


Caching intermediate results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ./caching.rst

Encoding class documentation standards
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ./class_documentation_standards.rst

Adding a unit test for the new encoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

to be added




