from pathlib import Path

import pandas as pd

from immuneML.data_model.datasets.ElementDataset import ReceptorDataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.PathBuilder import PathBuilder


class GLIPH2Exporter(DataReport):
    """
    Report which exports the receptor data to GLIPH2 format so that it can be directly used in GLIPH2 tool. Currently, the report accepts only
    receptor datasets.

    GLIPH2 publication: Huang H, Wang C, Rubelt F, Scriba TJ, Davis MM. Analyzing the Mycobacterium tuberculosis immune response by T-cell receptor
    clustering with GLIPH2 and genome-wide antigen screening. Nature Biotechnology. Published online April 27,
    2020:1-9. `doi:10.1038/s41587-020-0505-4 <https://www.nature.com/articles/s41587-020-0505-4>`_

    **Specification arguments:**

    - condition (str): name of the parameter present in the receptor metadata in the dataset; condition can be anything which can be processed in
      GLIPH2, such as tissue type or treatment.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_gliph2_exporter:
                    GLIPH2Exporter:
                        condition: epitope # for instance, epitope parameter is present in receptors' metadata with values such as "MtbLys" for Mycobacterium tuberculosis (as shown in the original paper).

    """

    @classmethod
    def build_object(cls, **kwargs):
        return GLIPH2Exporter(**kwargs)

    def __init__(self, dataset: ReceptorDataset = None, result_path: Path = None, name: str = None, condition: str = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.condition = condition

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        alpha_chains, beta_chains, trbv, trbj, subject_condition, count = [], [], [], [], [], []
        for index, receptor in enumerate(self.dataset.get_data()):
            alpha_chains.append(receptor.alpha.sequence_aa)
            beta_chains.append(receptor.beta.sequence_aa)
            trbv.append(receptor.beta.v_call)
            trbj.append(receptor.beta.j_call)
            subject_condition.append(f"{getattr(receptor.metadata, 'subject_id', str(index))}:{receptor.metadata[self.condition]}")
            count.append(receptor.beta.duplicate_count
                         if receptor.beta.duplicate_count is not None else 1)

        df = pd.DataFrame({"CDR3b": beta_chains, "TRBV": trbv, "TRBJ": trbj, "CDR3a": alpha_chains,
                           "subject:condition": subject_condition, "count": count})
        file_path = self.result_path / "exported_data.tsv"
        df.to_csv(file_path, sep="\t", index=False)

        return ReportResult(self.name,
                            info="Report which exports the receptor data to GLIPH2 format so that it can be directly "
                                 "used in GLIPH2 tool.",
                            output_tables=[ReportOutput(file_path, "exported data in GLIPH2 format")])

    def check_prerequisites(self):
        if isinstance(self.dataset, ReceptorDataset):
            return True
        else:
            return False
