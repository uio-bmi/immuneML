import pandas as pd


class DataReshaper:

    @staticmethod
    def reshape(dataset):
        """
        Takes a 2D matrix of values from the encoded data and reshapes it to long format,
        retaining the column and row annotations. This is for ease of use in plotting the data.
        It is suggested that some sort of filtering is done first, otherwise the memory usage may explode, as
        the resulting data frame is of shape
        (matrix.shape[0] * matrix.shape[1], labels.shape[0] + feature_annotations.shape[1] + 1)
        """

        row_annotations = pd.DataFrame(dataset.encoded_data.labels)
        row_annotations["example_id"] = dataset.encoded_data.example_ids

        column_annotations = dataset.encoded_data.feature_annotations
        column_annotations["feature"] = dataset.encoded_data.feature_names

        matrix = dataset.encoded_data.examples
        matrix_1d = matrix.A.ravel()

        column_annotations = pd.concat([column_annotations]*matrix.shape[0], ignore_index=True)
        row_annotations = pd.DataFrame(row_annotations.values.repeat(matrix.shape[1], axis=0), columns=row_annotations.columns)
        data = pd.concat([row_annotations.reset_index(drop=True), column_annotations.reset_index(drop=True), pd.DataFrame({"value": matrix_1d})], axis=1)

        for column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="ignore")

        return data
