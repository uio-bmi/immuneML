import copy
import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelBinarizer, normalize

from source.analysis.AxisType import AxisType
from source.analysis.criteria_matches.CriteriaMatcher import CriteriaMatcher
from source.analysis.data_manipulation.GroupSummarizationType import GroupSummarizationType
from source.analysis.data_manipulation.NormalizationType import NormalizationType
from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.environment.Constants import Constants


class DataSummarizer:

    @staticmethod
    def group_repertoires(dataset: Dataset, group_columns, group_summarization_type: GroupSummarizationType):
        """
        Takes an encoded dataset and groups together repertoires by either adding or averaging the feature values for
        all repertoires with the same values (or combination of values) for the specified group_columns.
        """
        dataset = copy.deepcopy(dataset)

        repertoire_annotations = pd.DataFrame(dataset.encoded_data.labels)

        concatenated = DataSummarizer.concatenate_columns(repertoire_annotations, group_columns)

        group_mask = DataSummarizer.create_group_mask(concatenated.values, group_summarization_type)
        groups = group_mask["groups"]
        mask = group_mask["mask"]

        repertoires = mask.T.dot(dataset.encoded_data.repertoires)
        labels = DataSummarizer.split_values(groups, group_columns)
        metadata_path = DataSummarizer.build_metadata_from_labels(dataset.metadata_path, labels)

        encoded = EncodedData(
            repertoires=repertoires,
            labels=labels.to_dict("list"),
            repertoire_ids=groups,
            feature_names=dataset.encoded_data.feature_names,
            feature_annotations=dataset.encoded_data.feature_annotations
        )

        result = Dataset(
            data=dataset.data,
            params=dataset.params,
            encoded_data=encoded,
            filenames=dataset.get_filenames(),
            identifier=dataset.id,
            metadata_path=metadata_path
        )

        return result

    @staticmethod
    def build_metadata_from_labels(old_metadata_path: str, labels: pd.DataFrame) -> str:
        if old_metadata_path:
            path = os.path.dirname(os.path.abspath(old_metadata_path)) + "_{}_{}.csv"\
                    .format(os.path.splitext(os.path.basename(old_metadata_path))[0], str(labels.columns).replace(",", "_"))
            labels.to_csv(path)
        else:
            path = None
        return path

    @staticmethod
    def group_features(dataset: Dataset, group_columns, group_summarization_type: GroupSummarizationType):
        """
        Takes an encoded dataset and groups together features by either adding or averaging the feature values for
        all features with the same values (or combination of values) for the specified group_columns.
        """
        dataset = copy.deepcopy(dataset)

        if group_summarization_type == GroupSummarizationType.NONZERO:
            dataset.encoded_data.repertoires.data[:] = 1

        feature_annotations = dataset.encoded_data.feature_annotations

        concatenated = DataSummarizer.concatenate_columns(feature_annotations, group_columns)

        group_mask = DataSummarizer.create_group_mask(concatenated.values, group_summarization_type)
        groups = group_mask["groups"]
        mask = group_mask["mask"]

        repertoires = dataset.encoded_data.repertoires.dot(mask)
        feature_annotations = DataSummarizer.split_values(groups, group_columns)

        encoded = EncodedData(
            repertoires=repertoires,
            labels=dataset.encoded_data.labels,
            repertoire_ids=dataset.encoded_data.repertoire_ids,
            feature_names=groups,
            feature_annotations=feature_annotations
        )

        result = Dataset(
            data=dataset.data,
            params=dataset.params,
            encoded_data=encoded,
            filenames=dataset.get_filenames(),
            identifier=dataset.id,
            metadata_path=dataset.metadata_path
        )

        return result

    @staticmethod
    def concatenate_columns(data: pd.DataFrame, columns) -> pd.Series:
        concatenated = data[columns].apply(lambda row: Constants.FEATURE_DELIMITER.join(row.values.astype(str)), axis=1)
        return concatenated

    @staticmethod
    def split_values(values, column_names) -> pd.DataFrame:
        data = pd.Series(values).str.split(Constants.FEATURE_DELIMITER, expand=True)
        data.columns = column_names
        return data

    @staticmethod
    def create_group_mask(values, group_summarization_type: GroupSummarizationType):
        lb = LabelBinarizer(sparse_output=True)

        mask = lb.fit_transform(values)
        mask = DataSummarizer.normalize_label_binarizer_result(mask, values)

        groups = lb.classes_

        if group_summarization_type == GroupSummarizationType.AVERAGE:
            mask = mask @ sparse.diags(1 / mask.sum(axis=0).A.ravel())

        result = {
            "mask": mask,
            "groups": groups
        }

        return result

    @staticmethod
    def normalize_label_binarizer_result(mask, values):
        # Logic to deal with the fact that if there are only 2 unique classes, the final one-hot encoded representation
        # only includes one column, which is problematic for matrix multiplication.
        if mask.shape[1] == 1 and np.unique(values).shape[0] != 1:
            mask = np.hstack((np.abs(mask[:, 0].A - 1), mask.A))
            mask = sparse.csr_matrix(mask)
        # Logic to deal with only 1 unique class
        elif mask.shape[1] == 1 and np.unique(values).shape[0] == 1:
            mask = sparse.csr_matrix(mask.A + 1)

        return mask

    @staticmethod
    def build_metadata(metadata_path: str, indices):
        if metadata_path:
            df = pd.read_csv(metadata_path, index_col=0).iloc[indices, :]
            path = os.path.dirname(os.path.abspath(metadata_path)) + "_{}_filtered.csv"\
                .format(os.path.splitext(os.path.basename(metadata_path))[0])
            df.to_csv(path)
        else:
            path = None
        return path

    @staticmethod
    def filter_repertoires(dataset: Dataset, criteria: dict):
        """
        Takes an encoded dataset and filters repertoires based on a given set of criteria. Only repertories meeting
        these criteria will be retained in the new dataset object.
        """
        dataset = copy.deepcopy(dataset)

        data = pd.DataFrame(dataset.encoded_data.labels)

        matcher = CriteriaMatcher()
        results = matcher.match(criteria=criteria, data=data)
        indices = np.where(np.array(results))[0]

        metadata_path = DataSummarizer.build_metadata(dataset.metadata_path, indices)
        labels = data.iloc[indices, :].to_dict("list")
        repertoires = dataset.encoded_data.repertoires[indices, :]
        filenames = [dataset.get_filenames()[i] for i in indices]
        repertoire_ids = [dataset.encoded_data.repertoire_ids[i] for i in indices]

        encoded = EncodedData(
            repertoires=repertoires,
            labels=labels,
            repertoire_ids=repertoire_ids,
            feature_names=dataset.encoded_data.feature_names,
            feature_annotations=dataset.encoded_data.feature_annotations
        )

        result = Dataset(
            data=dataset.data,
            params=dataset.params,
            encoded_data=encoded,
            filenames=filenames,
            identifier=dataset.id,
            metadata_path=metadata_path
        )

        return result

    @staticmethod
    def filter_features(dataset: Dataset, criteria: dict):
        """
        Takes an encoded dataset and filters features based on a given set of criteria. Only features meeting
        these criteria will be retained in the new dataset object.
        """
        dataset = copy.deepcopy(dataset)

        feature_annotations = dataset.encoded_data.feature_annotations

        matcher = CriteriaMatcher()
        results = matcher.match(criteria=criteria, data=feature_annotations)
        indices = np.where(np.array(results))[0]

        feature_annotations = feature_annotations.iloc[indices, :]
        repertoires = dataset.encoded_data.repertoires[:, indices]
        feature_names = [dataset.encoded_data.feature_names[i] for i in indices]

        encoded = EncodedData(
            repertoires=repertoires,
            labels=dataset.encoded_data.labels,
            repertoire_ids=dataset.encoded_data.repertoire_ids,
            feature_names=feature_names,
            feature_annotations=feature_annotations
        )

        result = Dataset(
            data=dataset.data,
            params=dataset.params,
            encoded_data=encoded,
            filenames=dataset.get_filenames(),
            identifier=dataset.id,
            metadata_path=dataset.metadata_path
        )

        return result

    @staticmethod
    def annotate_repertoires(dataset: Dataset, criteria: dict, name: str = "annotation"):
        """
        Takes an encoded dataset and adds a new label to the encoded_dataset with boolean values showing whether a
        repertoire matched the specified criteria or not.
        """
        dataset = copy.deepcopy(dataset)

        data = pd.DataFrame(dataset.encoded_data.labels)

        matcher = CriteriaMatcher()
        results = matcher.match(criteria=criteria, data=data)

        labels = dataset.encoded_data.labels
        labels[name] = np.array(results)

        encoded = EncodedData(
            repertoires=dataset.encoded_data.repertoires,
            labels=labels,
            repertoire_ids=dataset.encoded_data.repertoire_ids,
            feature_names=dataset.encoded_data.feature_names,
            feature_annotations=dataset.encoded_data.feature_annotations
        )

        result = Dataset(
            data=dataset.data,
            params=dataset.params,
            encoded_data=encoded,
            filenames=dataset.get_filenames(),
            identifier=dataset.id,
            metadata_path=dataset.metadata_path
        )

        return result

    @staticmethod
    def annotate_features(dataset: Dataset, criteria: dict, name: str = "annotation"):
        """
        Takes an encoded dataset and adds a new column to the feature_annotations with boolean values showing whether a
        feature matched the specified criteria or not.
        """
        dataset = copy.deepcopy(dataset)

        feature_annotations = dataset.encoded_data.feature_annotations

        matcher = CriteriaMatcher()
        results = matcher.match(criteria=criteria, data=feature_annotations)

        feature_annotations[name] = results

        encoded = EncodedData(
            repertoires=dataset.encoded_data.repertoires,
            labels=dataset.encoded_data.labels,
            repertoire_ids=dataset.encoded_data.repertoire_ids,
            feature_names=dataset.encoded_data.feature_names,
            feature_annotations=feature_annotations
        )

        result = Dataset(
            data=dataset.data,
            params=dataset.params,
            encoded_data=encoded,
            filenames=dataset.get_filenames(),
            identifier=dataset.id,
            metadata_path=dataset.metadata_path
        )

        return result

    @staticmethod
    def normalize_repertoires(dataset: Dataset, normalization_type: NormalizationType):
        assert normalization_type in [NormalizationType.L2, NormalizationType.RELATIVE_FREQUENCY, NormalizationType.BINARY]
        dataset = copy.deepcopy(dataset)
        repertoires = DataSummarizer.normalize_matrix(dataset.encoded_data.repertoires, normalization_type, AxisType.REPERTOIRES)
        encoded = EncodedData(
            repertoires=repertoires,
            labels=dataset.encoded_data.labels,
            repertoire_ids=dataset.encoded_data.repertoire_ids,
            feature_names=dataset.encoded_data.feature_names,
            feature_annotations=dataset.encoded_data.feature_annotations
        )
        result = Dataset(
            data=dataset.data,
            params=dataset.params,
            encoded_data=encoded,
            filenames=dataset.get_filenames(),
            identifier=dataset.id,
            metadata_path=dataset.metadata_path
        )
        return result

    @staticmethod
    def normalize_features(dataset: Dataset, normalization_type: NormalizationType):
        assert normalization_type in [NormalizationType.BINARY]
        dataset = copy.deepcopy(dataset)
        repertoires = DataSummarizer.normalize_matrix(dataset.encoded_data.repertoires, normalization_type, AxisType.FEATURES)
        encoded = EncodedData(
            repertoires=repertoires,
            labels=dataset.encoded_data.labels,
            repertoire_ids=dataset.encoded_data.repertoire_ids,
            feature_names=dataset.encoded_data.feature_names,
            feature_annotations=dataset.encoded_data.feature_annotations
        )
        result = Dataset(
            data=dataset.data,
            params=dataset.params,
            encoded_data=encoded,
            filenames=dataset.get_filenames(),
            identifier=dataset.id,
            metadata_path=dataset.metadata_path
        )
        return result

    @staticmethod
    def normalize_matrix(matrix: sparse.csr_matrix, normalization_type: NormalizationType, axis: AxisType):
        if normalization_type == NormalizationType.NONE:
            result = matrix
        else:
            normalization_method = getattr(DataSummarizer, "normalize_" + normalization_type.name.lower())
            result = normalization_method(matrix, axis)
        return result

    @staticmethod
    def normalize_relative_frequency(matrix: sparse.csr_matrix, axis: AxisType):
        if axis == AxisType.REPERTOIRES:
            return sparse.diags(1 / matrix.sum(axis=1).A.ravel()) @ matrix
        if axis == AxisType.FEATURES:
            return matrix @ sparse.diags(1 / matrix.sum(axis=0).A.ravel())

    @staticmethod
    def normalize_l2(matrix: sparse.csr_matrix, axis: AxisType):
        axis = 1 if axis == AxisType.REPERTOIRES else 0
        return normalize(matrix, axis=axis)

    @staticmethod
    def normalize_binary(matrix: sparse.csr_matrix, axis: AxisType):
        matrix.data[:] = 1
        return matrix
