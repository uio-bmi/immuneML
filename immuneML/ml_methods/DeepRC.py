import hashlib
import warnings
from pathlib import Path

import h5py
import numpy as np
import pkg_resources
import torch
import yaml
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.deeprc.DeepRCEncoder import DeepRCEncoder
from immuneML.environment.Label import Label
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.ml_methods.util.Util import Util
from immuneML.util.FilenameHandler import FilenameHandler
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReflectionHandler import ReflectionHandler


class DeepRC(MLMethod):
    """
    This classifier uses the DeepRC method for repertoire classification. The DeepRC ML method should be used in combination
    with the DeepRC encoder. Also consider using the :ref:`DeepRCMotifDiscovery` report for interpretability.

    Notes:

    - DeepRC uses PyTorch functionalities that depend on GPU. Therefore, DeepRC does not work on a CPU.

    - This wrapper around DeepRC currently only supports binary classification.

    Reference:
    Michael Widrich, Bernhard Schäfl, Milena Pavlović, Geir Kjetil Sandve, Sepp Hochreiter, Victor Greiff, Günter Klambauer
    ‘DeepRC: Immune repertoire classification with attention-based deep massive multiple instance learning’.
    bioRxiv preprint doi: `https://doi.org/10.1101/2020.04.12.038158 <https://doi.org/10.1101/2020.04.12.038158>`_


    Arguments:

        validation_part (float):  the part of the data that will be used for validation, the rest will be used for training.

        add_positional_information (bool): whether positional information should be included in the input features.

        kernel_size (int): the size of the 1D-CNN kernels.

        n_kernels (int): the number of 1D-CNN kernels in each layer.

        n_additional_convs (int): Number of additional 1D-CNN layers after first layer

        n_attention_network_layers (int): Number of attention layers to compute keys

        n_attention_network_units (int): Number of units in each attention layer

        n_output_network_units (int): Number of units in the output layer

        consider_seq_counts (bool): whether the input data should be scaled by the receptor sequence counts.

        sequence_reduction_fraction (float): Fraction of number of sequences to which to reduce the number of sequences per bag based on attention weights. Has to be in range [0,1].

        reduction_mb_size (int): Reduction of sequences per bag is performed using minibatches of reduction_mb_size` sequences to compute the attention weights.

        n_updates (int): Number of updates to train for

        n_torch_threads (int):  Number of parallel threads to allow PyTorch

        learning_rate (float): Learning rate for adam optimizer

        l1_weight_decay (float): l1 weight decay factor. l1 weight penalty will be added to loss, scaled by `l1_weight_decay`

        l2_weight_decay (float): l2 weight decay factor. l2 weight penalty will be added to loss, scaled by `l2_weight_decay`

        evaluate_at (int): Evaluate model on training and validation set every `evaluate_at` updates. This will also check for a new best model for early stopping.

        sample_n_sequences (int): Optional random sub-sampling of `sample_n_sequences` sequences per repertoire. Number of sequences per repertoire might be smaller than `sample_n_sequences` if repertoire is smaller or random indices have been drawn multiple times. If None, all sequences will be loaded for each repertoire.

        training_batch_size (int): Number of repertoires per minibatch during training.

        n_workers (int): Number of background processes to use for converting dataset to hdf5 container and training set data loader.

        pytorch_device_name (str): The name of the pytorch device to use. This name will be passed to  torch.device(self.pytorch_device_name). The default value is cuda:0

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_deeprc_method:
            DeepRC:
                validation_part: 0.2
                add_positional_information: True
                kernel_size: 9

    """

    def __init__(self, validation_part, add_positional_information, kernel_size, n_kernels,
                 n_additional_convs, n_attention_network_layers, n_attention_network_units, n_output_network_units,
                 consider_seq_counts, sequence_reduction_fraction, reduction_mb_size, n_updates, n_torch_threads,
                 learning_rate, l1_weight_decay, l2_weight_decay, evaluate_at, sample_n_sequences, training_batch_size, n_workers,
                 keep_dataset_in_ram, pytorch_device_name):
        super(DeepRC, self).__init__()

        if not ReflectionHandler.is_installed("deeprc"):
            raise RuntimeError(f"{DeepRC.__name__}: deeprc module is not installed. Please check the documentation at "
                               f"https://docs.immuneml.uio.no/installation/install_with_package_manager.html for instructions how to install it.")

        from deeprc.deeprc_binary.training import train
        self.training_function = train

        self.model = None
        self.result_path = None

        self.max_seq_len = None
        self.label = None

        self.keep_dataset_in_ram = keep_dataset_in_ram
        self.pytorch_device_name = pytorch_device_name
        self.pytorch_device = torch.device(self.pytorch_device_name)

        # ML model setting (not inherited from DeepRC code)
        self.validation_part = validation_part

        # DeepRC class settings:
        self.add_positional_information = add_positional_information
        self.n_input_features = 20 + 3 * self.add_positional_information
        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        self.n_additional_convs = n_additional_convs
        self.n_attention_network_layers = n_attention_network_layers
        self.n_attention_network_units = n_attention_network_units
        self.n_output_network_units = n_output_network_units
        self.consider_seq_counts = consider_seq_counts
        self.sequence_reduction_fraction = sequence_reduction_fraction
        self.reduction_mb_size = reduction_mb_size

        # train function settings:
        self.evaluate_at = evaluate_at
        self.n_updates = n_updates
        self.n_torch_threads = n_torch_threads
        self.learning_rate = learning_rate
        self.l1_weight_decay = l1_weight_decay
        self.l2_weight_decay = l2_weight_decay

        # Dataloader related settings:
        self.sample_n_sequences = sample_n_sequences
        self.training_batch_size = training_batch_size
        self.n_workers = n_workers

        self.feature_names = None

    def _metadata_to_hdf5(self, metadata_filepath: Path, label_name: str):
        from deeprc.deeprc_binary.dataset_converters import DatasetToHDF5

        hdf5_filepath = metadata_filepath.parent / f"{metadata_filepath.stem}.hdf5"
        converter = DatasetToHDF5(metadata_file=str(metadata_filepath),
                                  id_column=DeepRCEncoder.ID_COLUMN,
                                  single_class_label_columns=tuple([label_name]),
                                  sequence_column=DeepRCEncoder.SEQUENCE_COLUMN,
                                  sequence_counts_column=DeepRCEncoder.COUNTS_COLUMN,
                                  column_sep=DeepRCEncoder.SEP,
                                  filename_extension=f".{DeepRCEncoder.EXTENSION}",
                                  verbose=False)
        converter.save_data_to_file(output_file=str(hdf5_filepath), n_workers=self.n_workers)

        return hdf5_filepath

    def _load_dataset_in_ram(self, hdf5_filepath: Path):
        with h5py.File(str(hdf5_filepath), 'r') as hf:
            pre_loaded_hdf5_file = dict()
            pre_loaded_hdf5_file['seq_lens'] = hf['sampledata']['seq_lens'][:]
            pre_loaded_hdf5_file['counts_per_sequence'] = hf['sampledata']['counts_per_sequence'][:]
            pre_loaded_hdf5_file['amino_acid_sequences'] = hf['sampledata']['amino_acid_sequences'][:]

        return pre_loaded_hdf5_file

    def _get_train_val_indices(self, n_examples, classes):
        """splits the data to training and validation and attempts to preserve the class distribution if possible"""

        indices = np.arange(0, n_examples)
        train_indices, val_indices = train_test_split(indices, test_size=self.validation_part, shuffle=True, stratify=classes)

        return train_indices, val_indices

    def make_data_loader(self, hdf5_filepath: Path, pre_loaded_hdf5_file, indices, label_name, eval_only: bool, is_train: bool, n_workers=1):
        """
        Creates a pytorch dataloader using DeepRC's RepertoireDataReaderBinary

        :param hdf5_filepath: the path to the HDF5 file
        :param pre_loaded_hdf5_file: Optional: It is faster to load the hdf5 file into the RAM as dictionary instead
            of keeping it on the disk. `pre_loaded_hdf5_file` is the loaded hdf5 file as dictionary.
            If None, the hdf5 file will be read from the disk and consume less RAM.
        :param indices: indices of the subset of repertoires in the data that will be used for this dataset.
                If 'None', all repertoires will be used.
        :param label_name: the name of the label to be predicted
        :param eval_only: whether the dataloader will only be used for evaluation (no training).
                if false, sample_n_sequences can be set
        :param is_train: whether this is a dataloader for training data. If true, self.training_batch_size is used.
        :param n_workers: the number of workers used in torch.utils.data.DataLoader
        :return: a Pytorch dataloader
        """
        from deeprc.deeprc_binary.dataset_readers import RepertoireDataReaderBinary
        from deeprc.deeprc_binary.dataset_readers import no_stack_collate_fn

        sample_n_sequences = None if eval_only else self.sample_n_sequences
        training_batch_size = self.training_batch_size if is_train else 1

        positive_class = self.label.positive_class if self.label.positive_class is not None else self.label.values[0]

        dataset = RepertoireDataReaderBinary(
            hdf5_filepath=str(hdf5_filepath), set_inds=indices,
            sample_n_sequences=sample_n_sequences, target_label=label_name,
            true_class_label_value=positive_class,
            pre_loaded_hdf5_file=pre_loaded_hdf5_file,
            verbose=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=training_batch_size,
                                                 shuffle=True,
                                                 num_workers=n_workers,
                                                 collate_fn=no_stack_collate_fn)
        return dataloader

    def _prepare_caching_params(self, encoded_data: EncodedData, type: str, label_name: str):
        return (("metadata_filepath", str(encoded_data.info["metadata_filepath"])),
                ("y", hashlib.sha256(str(encoded_data.labels[label_name]).encode("utf-8")).hexdigest()),
                ("label_name", label_name),
                ("type", type),
                ("validation_part", self.validation_part),
                ("add_positional_information", self.add_positional_information),
                ("n_input_features", self.n_input_features),
                ("kernel_size", self.kernel_size),
                ("n_kernels", self.n_kernels),
                ("n_additional_convs", self.n_additional_convs),
                ("n_attention_network_layers", self.n_attention_network_layers),
                ("n_attention_network_units", self.n_attention_network_units),
                ("n_output_network_units", self.n_output_network_units),
                ("consider_seq_counts", self.consider_seq_counts),
                ("sequence_reduction_fraction", self.sequence_reduction_fraction),
                ("reduction_mb_size", self.reduction_mb_size),
                ("n_updates", self.n_updates),
                ("n_torch_threads", self.n_torch_threads),
                ("learning_rate", self.learning_rate),
                ("l1_weight_decay", self.l1_weight_decay),
                ("l2_weight_decay", self.l2_weight_decay),
                ("sample_n_sequences", self.sample_n_sequences),
                ("training_batch_size", self.training_batch_size),
                ("n_workers", self.n_workers),
                ("evaluate_at", self.evaluate_at),
                ("pytorch_device_name", self.pytorch_device_name))

    def fit(self, encoded_data: EncodedData, label: Label, cores_for_training: int = 2):
        self.feature_names = encoded_data.feature_names
        self.label = label
        self.model = CacheHandler.memo_by_params(self._prepare_caching_params(encoded_data, "fit", label.name),
                                                 lambda: self._fit(encoded_data, label.name, cores_for_training))

    def _fit(self, encoded_data: EncodedData, label_name: str, cores_for_training: int = 2):

        hdf5_filepath = self._metadata_to_hdf5(encoded_data.info["metadata_filepath"], label_name)
        pre_loaded_hdf5_file = self._load_dataset_in_ram(hdf5_filepath) if self.keep_dataset_in_ram else None

        train_indices, val_indices = self._get_train_val_indices(len(encoded_data.example_ids), encoded_data.labels[label_name])
        self.max_seq_len = encoded_data.info["max_sequence_length"]

        self._fit_for_label(hdf5_filepath, pre_loaded_hdf5_file, train_indices, val_indices, label_name, cores_for_training)

        return self.model

    def _fit_for_label(self, hdf5_filepath: Path, pre_loaded_hdf5_file, train_indices, val_indices, label_name: str, cores_for_training: int):
        from deeprc.deeprc_binary.architectures import DeepRC as DeepRCInternal

        train_dataloader = self.make_data_loader(hdf5_filepath, pre_loaded_hdf5_file, train_indices, label_name, eval_only=False, is_train=True,
                                                 n_workers=self.n_workers)
        train_eval_dataloader = self.make_data_loader(hdf5_filepath, pre_loaded_hdf5_file, train_indices, label_name, eval_only=True, is_train=True)
        val_eval_dataloader = self.make_data_loader(hdf5_filepath, pre_loaded_hdf5_file, val_indices, label_name, eval_only=True, is_train=False)

        self.model = DeepRCInternal(n_input_features=self.n_input_features, n_output_features=1, max_seq_len=self.max_seq_len,
                                    kernel_size=self.kernel_size, consider_seq_counts=self.consider_seq_counts,
                                    n_kernels=self.n_kernels, n_additional_convs=self.n_additional_convs,
                                    n_attention_network_layers=self.n_attention_network_layers,
                                    n_attention_network_units=self.n_attention_network_units,
                                    n_output_network_layers=0,
                                    n_output_network_units=self.n_output_network_units,
                                    add_positional_information=self.add_positional_information,
                                    sequence_reduction_fraction=self.sequence_reduction_fraction,
                                    reduction_mb_size=self.reduction_mb_size, device=self.pytorch_device)

        self.training_function(self.model, trainingset_dataloader=train_dataloader, trainingset_eval_dataloader=train_eval_dataloader,
                               validationset_eval_dataloader=val_eval_dataloader, results_directory=self.result_path / "deeprc_log",
                               n_updates=self.n_updates, num_torch_threads=self.n_torch_threads, learning_rate=self.learning_rate,
                               l1_weight_decay=self.l1_weight_decay, l2_weight_decay=self.l2_weight_decay,
                               show_progress=False, device=self.pytorch_device, evaluate_at=self.evaluate_at)

    def fit_by_cross_validation(self, encoded_data: EncodedData, number_of_splits: int = 5, label: Label = None, cores_for_training: int = -1,
                                optimization_metric=None):
        warnings.warn("DeepRC: cross-validation on this classifier is not defined: fitting one model instead...")
        self.fit(encoded_data, label)

    def get_params(self):
        return {name: param.data.tolist() for name, param in self.model.named_parameters()}

    def check_is_fitted(self, label_name: str):
        if label_name != self.label.name:
            raise NotFittedError("This DeepRCs instance is not fitted yet. "
                                 "Call 'fit' with appropriate arguments before using this method.")

    def predict(self, encoded_data: EncodedData, label: Label):
        probabilities = self.predict_proba(encoded_data, label)
        predictions = dict()

        classes = self.get_classes()
        pos_class_probs = probabilities[label.name][:, 0]
        predictions[label.name] = [classes[0] if probability > 0.5 else classes[1] for probability in pos_class_probs]

        if label.positive_class is not None:
            predictions[label.name] = [pred_class == label.positive_class for pred_class in predictions[label.name]]

        return predictions

    def predict_proba(self, encoded_data: EncodedData, label: Label):
        self.check_is_fitted(label.name)

        probabilities = {}

        hdf5_filepath = self._metadata_to_hdf5(encoded_data.info["metadata_filepath"], label.name)
        pre_loaded_hdf5_file = self._load_dataset_in_ram(hdf5_filepath) if self.keep_dataset_in_ram else None

        test_dataloader = self.make_data_loader(hdf5_filepath, pre_loaded_hdf5_file, indices=None, label_name=label.name, eval_only=True, is_train=False)

        probs_pos_class = self._model_predict(self.model, test_dataloader)
        probabilities[label.name] = np.vstack((probs_pos_class, 1 - probs_pos_class)).T

        return probabilities

    def _model_predict(self, model, dataloader):
        """Based on the DeepRC function evaluate (deeprc.deeprc_binary.training.evaluate)"""
        with torch.no_grad():
            model.to(device=self.pytorch_device)
            scoring_predictions = []
            for scoring_data in tqdm(dataloader, total=len(dataloader), desc="Evaluating model",
                                     disable=True, position=1):
                # Get samples as lists
                labels, inputs, sequence_lengths, counts_per_sequence, sample_ids = scoring_data

                # Apply attention-based sequence reduction and create minibatch
                labels, inputs, sequence_lengths, n_sequences = model.reduce_and_stack_minibatch(
                    labels, inputs, sequence_lengths, counts_per_sequence)

                # Compute predictions from reduced sequences
                logit_outputs = model(inputs, n_sequences)
                prediction = torch.sigmoid(logit_outputs)
                scoring_predictions.append(prediction)

            # predictions
            scoring_predictions = torch.cat(scoring_predictions, dim=0).float().cpu().numpy()

        return scoring_predictions

    def load(self, path: Path, details_path: Path = None):
        name = FilenameHandler.get_filename(self.__class__.__name__, "pt")
        file_path = path / name
        if file_path.is_file():
            self.model = torch.load(str(file_path))
            self.model.eval()
        else:
            raise FileNotFoundError(f"{self.__class__.__name__} model could not be loaded from {file_path}. "
                                    f"Check if the path to the {name} file is properly set.")

        if details_path is None:
            params_path = path / FilenameHandler.get_filename(self.__class__.__name__, "yaml")
        else:
            params_path = details_path

        if params_path.is_file():
            with params_path.open("r") as file:
                desc = yaml.safe_load(file)
                if "label" in desc:
                    setattr(self, "label", Label(**desc["label"]))
                for param in ["feature_names", "classes"]:
                    if param in desc:
                        setattr(self, param, desc[param])

    def store(self, path, feature_names=None, details_path: Path = None):
        PathBuilder.build(path)
        name = FilenameHandler.get_filename(self.__class__.__name__, "pt")
        torch.save(self.model, str(path / name))

        if details_path is None:
            params_path = path / FilenameHandler.get_filename(self.__class__.__name__, "yaml")
        else:
            params_path = details_path

        with params_path.open("w") as file:
            desc = {
                **(self.get_params()),
                "feature_names": feature_names,
                "classes": self.get_classes()
            }
            if self.label is not None:
                desc["label"] = vars(self.label)
            yaml.dump(desc, file)

    def check_if_exists(self, path):
        file_path = path / FilenameHandler.get_filename(self.__class__.__name__, "pt")

        return file_path.is_file()

    def get_package_info(self) -> str:
        return 'immuneML ' + Util.get_immuneML_version() + '; deepRC ' + pkg_resources.get_distribution('DeepRC').version

    def get_feature_names(self) -> list:
        return self.feature_names

    def can_predict_proba(self) -> bool:
        return True

    def get_class_mapping(self) -> dict:
        return {}

    def get_label_name(self) -> str:
        return self.label.name

    def get_compatible_encoders(self):
        return [DeepRCEncoder]
