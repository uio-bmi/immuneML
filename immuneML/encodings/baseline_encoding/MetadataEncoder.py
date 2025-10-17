import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer

from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.util.ParameterValidator import ParameterValidator


class MetadataEncoder(DatasetEncoder):
    """
    Encoder that uses metadata fields as features, such as HLA.

    **Dataset type:**
    - RepertoireDatasets
    - SequenceDatasets
    - ReceptorDatasets

    **Specification arguments:**

    - metadata_fields (list): List of metadata fields to use as features.

    **YAML specification:**

    .. code-block:: yaml

        encodings:
            metadata_encoding:
                Metadata:
                    metadata_fields: [HLA, sex]

    """

    def __init__(self, metadata_fields: list, name: str = None):
        super().__init__(name=name)
        self.metadata_fields = metadata_fields
        self.mlbs = None

    @staticmethod
    def build_object(dataset: Dataset, **params):
        assert 'metadata_fields' in params, "Parameter 'metadata_fields' is required for MetadataEncoder."
        ParameterValidator.assert_type_and_value(params['metadata_fields'], list, 'MetadataEncoder', 'metadata_fields')
        ParameterValidator.assert_all_type_and_value(params['metadata_fields'], str, 'MetadataEncoder', 'metadata_fields')

        ParameterValidator.assert_all_in_valid_list(params['metadata_fields'], dataset.get_label_names(),
                                                   'MetadataEncoder', 'metadata_fields')

        name = params.get('name', 'metadata_encoding')
        return MetadataEncoder(metadata_fields=params['metadata_fields'], name=name)

    def encode(self, dataset, params: EncoderParams) -> Dataset:

        metadata = dataset.get_metadata(self.metadata_fields, return_df=True)

        features = None
        classes = []

        if params.learn_model:
            self.mlbs = {feature: None for feature in self.metadata_fields}

        for feature in self.metadata_fields:
            flattened, mlb = flatten_comma_separated_mlb(metadata, feature, self.mlbs[feature])
            self.mlbs[feature] = mlb
            classes += [f"{feature}_{c}" for c in mlb.classes_.tolist()]
            if features is None:
                features = flattened
            else:
                features = pd.concat([features, flattened], axis=1)

        # Convert to sparse matrix
        onehot_sparse = csr_matrix(features)

        labels = {label: dataset.get_metadata([label])[label] for label in params.label_config.get_labels_by_name()} \
            if params.encode_labels else None

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(
            examples=onehot_sparse,
            feature_names=classes,
            feature_annotations=pd.DataFrame({"feature": classes}),
            labels=labels,
            info={'metadata_fields': self.metadata_fields}
        )

        return encoded_dataset


def flatten_comma_separated_mlb(df, column_name, mlb=None):
    """
    Flatten comma-separated values using MultiLabelBinarizer.

    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    column_name : str
        Name of column containing comma-separated values

    Returns:
    --------
    pandas DataFrame with original columns + one-hot encoded columns
    """

    # Split comma-separated values into lists
    split_values = df[column_name].str.split(',').tolist()

    # Remove whitespace from each element
    split_values = [[item.strip() for item in row if item.strip()] for row in split_values]

    # Create binary matrix
    if mlb is None:
        mlb = MultiLabelBinarizer()
        binary_matrix = mlb.fit_transform(split_values)
    else:
        binary_matrix = mlb.transform(split_values)

    # Create new column names
    new_columns = [f"{column_name}_{label}" for label in mlb.classes_]

    # Create DataFrame with binary columns
    binary_df = pd.DataFrame(binary_matrix, columns=new_columns, index=df.index)

    return binary_df, mlb
