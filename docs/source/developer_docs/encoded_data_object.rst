  **EncodedData:**

  - :code:`examples`: a design matrix where the rows represent Repertoires, Receptors or Sequences ('examples'), and the columns the encoding-specific features. This is typically a numpy matrix, but may also be another matrix type (e.g., scipy sparse matrix, pytorch tensor, pandas dataframe).
  - :code:`encoding`: a string denoting the encoder base class that was used.
  - :code:`labels`: a dictionary of labels, where each label is a key, and the values are the label values across the examples (for example: {disease1: [positive, positive, negative]} if there are 3 repertoires). This parameter should be set only if :code:`EncoderParams.encode_labels` is True, otherwise it should be set to None. This can be created by calling utility function :code:`EncoderHelper.encode_dataset_labels()`.
  - :code:`example_ids`: a list of identifiers for the examples (Repertoires, Receptors or Sequences). This can be retrieved using :code:`Dataset.get_example_ids()`.
  - :code:`feature_names`: a list of feature names, i.e., the names given to the encoding-specific features. When included, list must be as long as the number of features.
  - :code:`feature_annotations`: an optional pandas dataframe with additional information about the features. When included, number of rows in this dataframe must correspond to the number of features. This parameter is not typically used.
  - :code:`info`: an optional dictionary that may be used to store any additional information that is relevant (for example paths to additional output files). This parameter is not typically used.
