import warnings

import numpy as np

from immuneML.ml_methods.DeepRC import DeepRC
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class DeepRCMotifDiscovery(MLReport):
    """
    This report plots the contributions of (i) input sequences and (ii) kernels to trained DeepRC model with respect to
    the test dataset. Contributions are computed using integrated gradients (IG).
    This report produces two figures:
        - inputs_integrated_gradients: Shows the contributions of the characters within the input sequences (test dataset) that was most important for immune status prediction of the repertoire. IG is only applied to sequences of positive class repertoires.
        - kernel_integrated_gradients: Shows the 1D CNN kernels with the highest contribution over all positions and amino acids.

    For both inputs and kernels: Larger characters in the extracted motifs indicate higher contribution, with blue
    indicating positive contribution and red indicating negative contribution towards the prediction of the immune status.
    For kernels only: contributions to positional encoding are indicated by < (beginning of sequence),
    ∧ (center of sequence), and > (end of sequence).

    Reference:
    Michael Widrich, Bernhard Schäfl, Milena Pavlović, Geir Kjetil Sandve, Sepp Hochreiter, Victor Greiff, Günter Klambauer
    ‘DeepRC: Immune repertoire classification with attention-based deep massive multiple instance learning’.
    bioRxiv preprint doi: `https://doi.org/10.1101/2020.04.12.03815 <https://doi.org/10.1101/2020.04.12.038158>`_


    Arguments:

        n_steps (int): Number of IG steps (more steps -> better path integral -> finer contribution values). 50 is usually good enough.

        threshold (float): Only applies to the plotting of kernels. Contributions are normalized to range [0, 1], and only kernels with normalized contributions above threshold are plotted.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_deeprc_report:
            DeepRCMotifDiscovery:
                threshold: 0.5
                n_steps: 50

    """

    def __init__(self, n_steps, threshold, name: str = None):
        super().__init__(name=name)
        self.n_steps = n_steps
        self.threshold = threshold
        self.filename_inputs = "inputs_integrated_gradients.pdf"
        self.filename_kernels = "kernel_integrated_gradients.pdf"
        self.name = name

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
        label_names = [self.label]
        hdf5_filepath = self.method._metadata_to_hdf5(test_metadata_filepath, label_names)

        n_examples_test = len(self.test_dataset.encoded_data.example_ids)
        indices = np.array(range(n_examples_test))

        dataloader = self.method.make_data_loader(hdf5_filepath, pre_loaded_hdf5_file=None,
                                                  indices=indices, label=self.label, eval_only=True,
                                                  is_train=False)

        model = self.method.get_model(self.label)[self.label]

        compute_contributions(intgrds_set_loader=dataloader, deeprc_model=model, n_steps=self.n_steps,
                              threshold=self.threshold, resdir=self.result_path, filename_inputs=self.filename_inputs,
                              filename_kernels=self.filename_kernels)

        return ReportResult(self.name,
                            output_figures=[ReportOutput(self.filename_inputs),
                                            ReportOutput(self.filename_kernels)])

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
