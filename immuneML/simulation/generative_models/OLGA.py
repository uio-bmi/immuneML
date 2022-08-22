from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from olga import load_model
from olga.sequence_generation import SequenceGenerationVJ, SequenceGenerationVDJ

from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.IO.dataset_import.OLGAImport import OLGAImport
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.generative_models.GenerativeModel import GenerativeModel
from immuneML.util.ImportHelper import ImportHelper
from immuneML.util.ParameterValidator import ParameterValidator


@dataclass
class OLGA(GenerativeModel):
    """
    This class is an immuneML wrapper around OLGA package as described by Sethna et al. 2019 that is available in `olga` python package
    available at PyPI and at the following GitHub repository: https://github.com/statbiophys/OLGA.

    Reference:

    Zachary Sethna, Yuval Elhanati, Curtis G Callan, Jr, Aleksandra M Walczak, Thierry Mora, OLGA: fast computation of generation probabilities
    of B- and T-cell receptor amino acid sequences and motifs, Bioinformatics, Volume 35, Issue 17, 1 September 2019, Pages 2974–2981,
    https://doi.org/10.1093/bioinformatics/btz035

    """
    model_path: Path
    default_model_name: str
    chain: Chain = None
    use_only_productive: bool = True
    sequence_gen_model: Union[SequenceGenerationVDJ, SequenceGenerationVJ] = None
    v_gene_mapping: np.ndarray = None
    j_gene_mapping: np.ndarray = None

    DEFAULT_MODEL_FOLDER_MAP = {
        "humanTRA": "human_T_alpha", "humanTRB": "human_T_beta",
        "humanIGH": "human_B_heavy", "humanIGK": "human_B_kappa", "humanIGL": "human_B_lambda",
        "mouseTRB": "mouse_T_beta", "mouseTRA": "mouse_T_alpha"
    }
    MODEL_FILENAMES = {'marginals': 'model_marginals.txt', 'params': 'model_params.txt', 'v_gene_anchor': 'V_gene_CDR3_anchors.csv',
                       'j_gene_anchor': 'J_gene_CDR3_anchors.csv'}
    OUTPUT_COLUMNS = ["sequence", 'sequence_aa', 'v_call', 'j_call', 'region_type', "frame_type"]

    @classmethod
    def build_object(cls, **kwargs):

        location = OLGA.__name__

        ParameterValidator.assert_keys(kwargs.keys(), ['model_path', 'default_model_name', 'chain', 'use_only_productive'], location, 'OLGA generative model')
        ParameterValidator.assert_type_and_value(kwargs['use_only_productive'], bool, location, 'use_only_productive')

        if kwargs['model_path']:
            assert Path(kwargs['model_path']).is_dir(), \
                f"{OLGA.__name__}: the model path is not a directory. It has to be a directory and contain files with the exact names as " \
                f"described in the OLGA package documentation: https://github.com/statbiophys/OLGA."

            for filename in OLGA.MODEL_FILENAMES.values():
                assert (Path(kwargs['model_path']) / filename).is_file(), \
                    f"{OLGA.__name__}: file {filename} is missing in the specified directory: {kwargs['model_path']}"

            assert kwargs['default_model_name'] is None, \
                f"{OLGA.__name__}: default_model_name must be None when model_path is set, but now it is {kwargs['default_model_name']}."
            chain = Chain.get_chain(kwargs['chain'])
        else:
            ParameterValidator.assert_in_valid_list(kwargs['default_model_name'], list(OLGA.DEFAULT_MODEL_FOLDER_MAP.keys()), location,
                                                    'default_model_name')
            chain = Chain.get_chain(kwargs['default_model_name'][-3:])
            kwargs['model_path'] = Path(load_model.__file__).parent / f"default_models/{OLGA.DEFAULT_MODEL_FOLDER_MAP[kwargs['default_model_name']]}"

        return OLGA(**{**kwargs, **{'chain': chain}})

    def load_model(self):
        is_vdj = self.chain in [Chain.BETA, Chain.HEAVY]
        olga_gen_model = load_model.GenerativeModelVDJ() if is_vdj else load_model.GenerativeModelVJ()
        olga_gen_model.load_and_process_igor_model(str(self.model_path / OLGA.MODEL_FILENAMES['marginals']))

        genomic_data = load_model.GenomicDataVDJ() if is_vdj else load_model.GenomicDataVJ()
        genomic_data.load_igor_genomic_data(params_file_name=str(self.model_path / OLGA.MODEL_FILENAMES['params']),
                                            V_anchor_pos_file=str(self.model_path / OLGA.MODEL_FILENAMES['v_gene_anchor']),
                                            J_anchor_pos_file=str(self.model_path / OLGA.MODEL_FILENAMES['j_gene_anchor']))

        self.v_gene_mapping = pd.read_csv(self.model_path / OLGA.MODEL_FILENAMES['v_gene_anchor'])['gene'].values
        self.j_gene_mapping = pd.read_csv(self.model_path / OLGA.MODEL_FILENAMES['j_gene_anchor'])['gene'].values

        self.sequence_gen_model = SequenceGenerationVDJ(olga_gen_model, genomic_data) if is_vdj \
            else SequenceGenerationVJ(olga_gen_model, genomic_data)

    def generate_sequences(self, count: int, seed: int = 1, path: Path = None, sequence_type: SequenceType = SequenceType.AMINO_ACID) -> pd.DataFrame:

        if not self.use_only_productive:
            raise NotImplementedError("Generating unproductive sequences is currently not supported.")

        if not self.sequence_gen_model:
            self.load_model()

        sequences = pd.DataFrame(index=np.arange(count), columns=OLGA.OUTPUT_COLUMNS)
        for i in range(count):
            seq_row = self.sequence_gen_model.gen_rnd_prod_CDR3()
            sequences.loc[i] = (seq_row[0], seq_row[1], self.v_gene_mapping[seq_row[2]], self.j_gene_mapping[seq_row[3]],
                                RegionType.IMGT_JUNCTION.name, SequenceFrameType.IN.name)

        return sequences

    def compute_p_gens(self):
        raise NotImplementedError

    def can_compute_p_gens(self) -> bool:
        return True

    def _import_olga_sequences(self, sequence_type: SequenceType, path: Path):
        import_empty_nt_sequences = False if sequence_type == SequenceType.NUCLEOTIDE else True

        default_params = DefaultParamsLoader.load('datasets', 'olga')
        params = DatasetImportParams.build_object(**{**default_params, **{'path': path, "import_empty_nt_sequences": import_empty_nt_sequences}})
        sequences = ImportHelper.import_items(OLGAImport, path, params)
        return sequences