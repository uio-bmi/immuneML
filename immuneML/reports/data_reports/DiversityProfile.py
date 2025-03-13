from pathlib import Path

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset, SequenceDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.evenness_profile.EvennessProfileEncoder import EvennessProfileEncoder
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.PathBuilder import PathBuilder


class DiversityProfile(DataReport):
    """
    Plots the diversity profiles of Repertoires in a Dataset
    The profiles are internally calculated using :py:obj:`~immuneML.encodings.evenness_profile.EvennessProfileEncoderÏ.EvennessProfileEncoder`

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_diversity_profiles:
                    DiversityProfile:
                        min_alpha: 0
                        max_alpha: 10
                        dimension: 51

    """
    UNKNOWN_CHAIN = "unknown"

    def __init__(self, dataset: Dataset = None, result_path: Path = None, number_of_processes: int = 1, name: str = None,
                 min_alpha: float = None, max_alpha: float = None, dimension: int = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.dimension = dimension

    @classmethod
    def build_object(cls, **kwargs):
        return DiversityProfile(**kwargs)

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)

        encoded_data = self._encode_repertoires()


        return ReportResult(name=self.name,
                            info=f"A simple overview of the properties of dataset {self.dataset.name}")

    def _encode_repertoires(self):
        encoder = EvennessProfileEncoder.build_object(self.dataset, **{
            "min_alpha": self.min_alpha,
            "max_alpha": self.max_alpha,
            "dimension": self.dimension
        })

        encoded_dataset = encoder.encode(self.dataset, EncoderParams(result_path=self.result_path / "encoded_dataset",
                                                                     label_config=LabelConfiguration(),
                                                                     pool_size=self.number_of_processes))

        return encoded_dataset.encoded_data

    def check_prerequisites(self):
        if isinstance(self.dataset, RepertoireDataset):
            return True
        else:
            return False
