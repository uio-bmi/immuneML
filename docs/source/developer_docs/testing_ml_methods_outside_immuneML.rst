Testing the ML method outside immuneML with a sample design matrix
-------------------------------------------------------------------

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML dev docs: testing the ML method outside immuneML with a sample design matrix
   :twitter:image: https://docs.immuneml.uio.no/_images/extending_immuneML.png

When implementing a new ML method, it can be useful to test the method with a small sample design matrix before integrating it into the immuneML
codebase. Example design matrices for any encoding can be exported to .csv format with the DesignMatrixExporter report and the ExploratoryAnalysis
instruction. To quickly generate some random sample data, RandomRepertoireDataset or RandomReceptorDataset may be specified as import formats.
Alternatively, you can import your own data. A full YAML specification for exporting a sample design matrix for a 3-mer encoding may look like this:

.. indent with spaces
.. code-block:: yaml

  definitions:
    datasets:
      my_simulated_data:
        format: RandomRepertoireDataset
        params:
          repertoire_count: 5 # a dataset with 5 repertoires
          sequence_count_probabilities: # each repertoire has 10 sequences
            10: 1
          sequence_length_probabilities: # each sequence has length 15
            15: 1
          labels:
            my_label: # half of the repertoires has my_label = true, the rest has false
              false: 0.5
              true: 0.5
    encodings:
      my_3mer_encoding:
        KmerFrequency:
          k: 3
    reports:
      my_design_matrix:
        DesignMatrixExporter:
          name: my_design_matrix
  instructions:
    my_instruction:
      type: ExploratoryAnalysis
      analyses:
        my_analysis:
          dataset: my_simulated_data
          encoding: my_3mer_encoding
          labels:
          - my_label
          report: my_design_matrix

.. note::

  Note that for design matrices beyond 2 dimensions (such as OneHotEncoder with flatten = False), the matrix is exported as a .npy file instead of a
  .csv file.

To generate the design matrix, save the YAML specification to specs.yaml and and run immuneML providing the path to the saved file and a path to the
output directory:

.. code-block:: console

  immune-ml specs.yaml output_dir/

The resulting design matrix can be found in my_instruction/analysis_my_analysis/report/design_matrix.csv, and the true classes for each repertoire
can be found in labels.csv. In immuneML, the design matrix is passed to the ML method as an EncodedData object, and the labels as a numpy ndarray.
The EncodedData object has attribute examples which contains the design matrix, and feature_names and example_ids which contain the row and column
names respectively.

Testing the ML method using immuneML for encoding the data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In immuneML, different encodings can be used to create encoded data. For the full list of possible encodings, see :ref:`Encodings`. Encoding works in the following
way: given a dataset and encoding parameters, the specific encoder object creates an instance of EncodedData class and fills the attributes examples
(design / feature matrix, where one row is one example), labels (a dictionary of numpy array where the keys are label names and values are arrays
with class assignment for each example), feature_names (if available for the encoding, a list of feature names for each column in the examples).
This object will be provided as input to the corresponding functions of the new ML method class.

To load the data encoded as described above into an EncodedData object, the function :py:obj:`immuneML.dev_util.util.load_encoded_data` can be used.