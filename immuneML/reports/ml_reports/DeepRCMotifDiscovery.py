import warnings
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.ml_methods.classifiers.DeepRC import DeepRC
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class DeepRCMotifDiscovery(MLReport):
    """
    This report plots the contributions of (i) input sequences and (ii) kernels to trained :ref:`DeepRC` model with respect to
    the test dataset. Contributions are computed using integrated gradients (IG).
    This report produces two figures:

    - inputs_integrated_gradients: Shows the contributions of the characters within the input sequences (test dataset) that was most important for immune status prediction of the repertoire. IG is only applied to sequences of positive class repertoires.
    - kernel_integrated_gradients: Shows the 1D CNN kernels with the highest contribution over all positions and amino acids.

    For both inputs and kernels: Larger characters in the extracted motifs indicate higher contribution, with blue
    indicating positive contribution and red indicating negative contribution towards the prediction of the immune status.
    For kernels only: contributions to positional encoding are indicated by < (beginning of sequence),
    ∧ (center of sequence), and > (end of sequence).

    See :ref:`DeepRCMotifDiscovery for repertoire classification` for a usage example.

    Reference:

    Widrich, M., et al. (2020). Modern Hopfield Networks and Attention for Immune Repertoire Classification. Advances in
    Neural Information Processing Systems, 33. https://proceedings.neurips.cc//paper/2020/hash/da4902cb0bc38210839714ebdcf0efc3-Abstract.html


    Specification arguments:

    - n_steps (int): Number of IG steps (more steps -> better path integral -> finer contribution values). 50 is usually good enough.

    - threshold (float): Only applies to the plotting of kernels. Contributions are normalized to range [0, 1], and only kernels with normalized contributions above threshold are plotted.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_deeprc_report:
            DeepRCMotifDiscovery:
                threshold: 0.5
                n_steps: 50

    """

    def __init__(self, n_steps, threshold, train_dataset: Dataset = None, test_dataset: Dataset = None,
                 method: MLMethod = None, result_path: Path = None, name: str = None, hp_setting: HPSetting = None,
                 label=None, number_of_processes: int = 1):
        super().__init__(train_dataset=train_dataset, test_dataset=test_dataset, method=method, result_path=result_path,
                         name=name, hp_setting=hp_setting, label=label, number_of_processes=number_of_processes)
        self.n_steps = n_steps
        self.threshold = threshold
        self.filename_inputs = Path("inputs_integrated_gradients.pdf")
        self.filename_kernels = Path("kernel_integrated_gradients.pdf")

    @classmethod
    def build_object(cls, **kwargs):
        location = "DeepRCMotifDiscovery"
        name = kwargs["name"] if "name" in kwargs else None
        ParameterValidator.assert_type_and_value(kwargs["n_steps"], int, location, "n_steps", min_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["threshold"], float, location, "threshold", min_inclusive=0, max_inclusive=1)

        return DeepRCMotifDiscovery(n_steps=kwargs["n_steps"], threshold=kwargs["threshold"], name=name)

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)

        test_metadata_filepath = self.test_dataset.encoded_data.info['metadata_filepath']
        hdf5_filepath = self.method._metadata_to_hdf5(metadata_filepath=test_metadata_filepath, label_name=self.label.name)

        n_examples_test = len(self.test_dataset.encoded_data.example_ids)
        indices = np.array(range(n_examples_test))

        dataloader = self.method.make_data_loader(hdf5_filepath, pre_loaded_hdf5_file=None,
                                                  indices=indices, label_name=self.label.name, eval_only=True,
                                                  is_train=False)

        path_inputs = self.result_path / self.filename_inputs
        path_kernels = self.result_path / self.filename_kernels

        self.compute_contributions(intgrds_set_loader=dataloader, deeprc_model=self.method.model, n_steps=self.n_steps,
                                   threshold=self.threshold, path_inputs=path_inputs,
                                   path_kernels=self.result_path / self.filename_kernels)

        return ReportResult(self.name,
                            info="Plots the contributions of (i) input sequences and (ii) kernels to trained `DeepRC` model with respect to the test dataset. Contributions are computed using integrated gradients.",
                            output_figures=[ReportOutput(path_inputs, "Integrated Gradients over the inputs to DeepRC"),
                                            ReportOutput(path_kernels, "Integrated Gradients over the kernels of DeepRC")])

    def compute_contributions(self, intgrds_set_loader: torch.utils.data.DataLoader, deeprc_model,
                              n_steps: int = 50, threshold: float = 0.5, path_inputs: Path = Path("inputs_integrated_gradients.pdf"),
                              path_kernels: Path = Path("kernel_integrated_gradients.pdf")):
        """ Compute and plot contributions of sequences and motifs to trained DeepRC model, given a dataset.
        Contribution is computed using integrated gradients (IG).

        Author -- Michael Widrich
        Created on -- 2020-07-20
        Contact -- michael.widrich@jku.at

        Parameters
        ----------
        intgrds_set_loader : torch.utils.data.DataLoader
            The dataset to compute IG for in form of a PyTorch DataLoader following the DeepRC format.
            E.g. one of the dataloaders returned by deeprc.deeprc_binary.predefined_datasets.cmv_dataset().
        deeprc_model : deeprc.deeprc_binary.architectures.DeepRC
            DeepRC model to compute IG for.
            Weights of first CNN layer are accessed via deeprc_model.sequence_embedding_16bit.conv_aas.weight .
        n_steps : int
            Number of IG steps (more steps -> better path integral -> finer contribution values). 50 is usually good enough.
        threshold : float
            Threshold for plotting of kernels (=motifs). Contributions are normalized to range [0, 1] and then threshold
            is applied. 0.5 -> only kernels with normalized contributions above 0.5 are plotted.
        path_inputs : Path
            path for inputs integrated gradients plot
        path_kernels : Path
            path for kernels integrated gradients plot
        """
        from tqdm import tqdm

        intgrds_set = intgrds_set_loader.dataset

        # Integrated gradients w.r.t. kernels
        active_kernels = deeprc_model.sequence_embedding_16bit.conv_aas.weight
        original_kernel_values = active_kernels.cpu().clone().data.detach().to(dtype=torch.float32)

        # Compute IG w.r.t. kernels -> prepare array for IG values
        int_grd_kernels = np.zeros(original_kernel_values.shape, dtype=np.float)

        # Compute interpolated kernels
        interp_factors = torch.linspace(1, 0, n_steps)
        interpolated_kernels = [original_kernel_values * intf for intf in interp_factors]
        most_important_inputs = []
        most_important_inputs_intgrds = []
        most_important_inputs_lens = []

        for data_index, data in tqdm(enumerate(intgrds_set_loader), total=len(intgrds_set_loader),
                                     desc="calc intgrds", ncols=10, position=0):

            # Get 1 sample as lists
            labels, inputs, sequence_lengths, duplicates_per_sequence, sample_ids = data

            # Only consider positive samples
            if labels[0][0]:
                continue

            # Perform attention pooling
            with torch.no_grad():
                labels, inputs, sequence_lengths, n_sequences = deeprc_model.reduce_and_stack_minibatch(
                    labels, inputs, sequence_lengths, duplicates_per_sequence)

            # Integrated gradients for kernels
            for step in tqdm(range(n_steps), total=n_steps, desc="kernels", ncols=10, position=1):
                deeprc_model.zero_grad()
                with torch.no_grad():
                    active_kernels.data[:] = interpolated_kernels[step].data.to(dtype=torch.float16)

                logit_outputs = deeprc_model(inputs, n_sequences)
                # prediction = torch.sigmoid(logit_outputs)
                logit_outputs.backward()
                int_grd_kernels[:] += (
                    original_kernel_values * active_kernels.grad.to(device='cpu', dtype=torch.float32)
                    / n_steps / len(intgrds_set_loader)).detach().data.cpu().numpy()

            # Integrated gradients for inputs
            with torch.no_grad():
                active_kernels.data[:] = original_kernel_values.data.to(dtype=torch.float16)
            interpolated_inputs = [inputs * intf for intf in interp_factors]
            int_grd_inputs = np.zeros(inputs.shape, dtype=np.float)
            inputs_cpu = inputs.to(device='cpu', dtype=torch.float32)
            for step in tqdm(range(n_steps), total=n_steps, desc="inputs", ncols=10, position=1):
                deeprc_model.zero_grad()
                i = torch.tensor(interpolated_inputs[step], requires_grad=True)
                i.requires_grad_()
                i.retain_grad()
                logit_outputs = deeprc_model(i, n_sequences)
                int_grd_inputs[:] += (
                    inputs_cpu * torch.autograd.grad(logit_outputs, i, retain_graph=True)[0].to(device='cpu',
                                                                                                dtype=torch.float32)
                    / n_steps).detach().data.cpu().numpy()
            most_important_input_ind = np.argmax(np.sum(np.sum(int_grd_inputs, 1), 1))
            most_important_inputs_intgrds.append(int_grd_inputs[most_important_input_ind, :, :].sum(axis=0))
            most_important_inputs.append(inputs_cpu[most_important_input_ind].detach().cpu().numpy())
            most_important_inputs_lens.append(sequence_lengths[most_important_input_ind].detach().cpu().numpy())

        # Plot inputs
        abs_max = np.max([np.abs(i).max() for i in most_important_inputs_intgrds])
        for i in range(len(most_important_inputs_intgrds)):
            most_important_inputs_intgrds[i][:] /= abs_max
            most_important_inputs[i] = intgrds_set.inds_to_aa(np.argmax(most_important_inputs[i][:-3], axis=0))

        self.plot_inputs_text(
            chars=most_important_inputs,
            colorgrad=most_important_inputs_intgrds,
            seq_lens=most_important_inputs_lens,
            file_path=path_inputs)

        # Get/Plot kernels with highest contribution over all positions and AAs
        kernel_contrib = int_grd_kernels.sum(axis=2).sum(axis=1)
        normed_kernel_contrib = kernel_contrib / kernel_contrib.max()
        n_top_kernels = np.sum(normed_kernel_contrib > threshold)

        top_kernel_inds = normed_kernel_contrib.argsort()[::-1]
        top_kernel_inds = top_kernel_inds[:n_top_kernels]
        top_kernels = int_grd_kernels[top_kernel_inds]
        top_kernels /= kernel_contrib.max()

        self.plot_kernels_text(kernels=top_kernels, charset=intgrds_set.aas, file_path=path_kernels)

    def plot_inputs_text(self, chars, colorgrad, seq_lens, file_path):
        """
        Author -- Michael Widrich
        Created on -- 2020-07-20
        Contact -- michael.widrich@jku.at
        """
        char_scale = 10
        char_offset = 1
        max_seq_len = max(seq_lens) + 1

        n_seqs = len(chars)
        fig = plt.figure(figsize=(int(np.round(max_seq_len / 15)), max(int(np.round(n_seqs / 5)), 5)))
        # fig = plt.figure(figsize=(int(np.round(max_seq_len)), max(int(np.round(n_seqs)), 5)))

        _ = [([fig.text((char_i + 1) / max_seq_len, 0 + (1 - (seq_i / (n_seqs))) - 0.05,
                        chars[seq_i][char_i].decode("utf-8"),
                        size=char_offset + abs(colorgrad[seq_i][char_i]) * char_scale,
                        ha='center', va='center',
                        color='blue' if colorgrad[seq_i][char_i] > 0 else 'red')
               for char_i in range(int(seq_lens[seq_i]))], None)
             for seq_i in range(n_seqs)]

        fig.savefig(str(file_path))
        plt.close(fig)
        del fig

    def plot_kernels_text(self, kernels, charset, file_path):
        """
        Author -- Michael Widrich
        Created on -- 2020-07-20
        Contact -- michael.widrich@jku.at
        """
        char_scale = 100
        char_offset = 1
        max_n_kernels = 15
        max_n_kernel_features = 15
        kernels = kernels[:max_n_kernels, ..., :max_n_kernel_features]

        n_kernels = kernels.shape[0]
        if not n_kernels:
            return None

        fig, axes = plt.subplots(1, n_kernels,
                                 figsize=(int(np.round(kernels.shape[-1] * kernels.shape[0] / 3)),
                                          int(np.round(len(charset) / 3))))
        if isinstance(axes, np.ndarray):
            axes = list(axes)
            if isinstance(axes[0], np.ndarray):
                axes = [list(a) for a in axes]
        if not isinstance(axes, list):
            axes = [axes]

        for a_i, ax in enumerate(axes):
            ax.axis('off')
            for out_i in range(kernels.shape[-1]):
                sorted_charset = [(char, char_offset + abs(char_contrib) * char_scale,
                                   'blue' if kernels[a_i, char_i, out_i] > 0 else 'red')
                                  for char_i, char, char_contrib
                                  in zip(range(len(charset)), charset, kernels[a_i, :, out_i])]
                sorted_charset.sort(key=lambda x: x[1])
                _ = [ax.text(out_i / kernels.shape[-1], 0 + (1 - (char_i / len(charset))), char,
                             size=char_size, ha='center', va='center', color=color)
                     for char_i, (char, char_size, color) in enumerate(sorted_charset)]

        fig.savefig(str(file_path))
        plt.close(fig)
        del fig

    def check_prerequisites(self):
        run_report = True

        if not hasattr(self, "result_path") or self.result_path is None:
            warnings.warn(f"{self.__class__.__name__} requires an output 'path' to be set. {self.__class__.__name__} report will not be created.")
            run_report = False

        if not isinstance(self.method, DeepRC):
            warnings.warn(
                f"{self.__class__.__name__} can only be used in combination with the DeepRC ML method. {self.__class__.__name__} report will not be created.")
            run_report = False

        if self.test_dataset.encoded_data is None:
            warnings.warn(
                f"{self.__class__.__name__}: test dataset is not encoded and can not be run. "
                f"{self.__class__.__name__} report will not be created.")
            run_report = False

        return run_report
