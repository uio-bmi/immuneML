import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from bionumpy.bnpdataclass import BNPDataClass
from olga import load_model
from olga.generation_probability import GenerationProbabilityVJ, GenerationProbabilityVDJ
from olga.sequence_generation import SequenceGenerationVJ, SequenceGenerationVDJ

from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.IO.dataset_import.OLGAImport import OLGAImport
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.generative_models.GenerativeModel import GenerativeModel
from immuneML.simulation.generative_models.InternalOlgaModel import InternalOlgaModel
from immuneML.simulation.util.igor_helper import make_skewed_model_files
from immuneML.util.ImportHelper import ImportHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


@dataclass
class OLGA(GenerativeModel):
    """
    This class is an immuneML wrapper around OLGA package as described by Sethna et al. 2019 that is available in `olga` python package
    available at PyPI and at the following GitHub repository: https://github.com/statbiophys/OLGA.

    Reference:

    Zachary Sethna, Yuval Elhanati, Curtis G Callan, Jr, Aleksandra M Walczak, Thierry Mora, OLGA: fast computation of generation probabilities
    of B- and T-cell receptor amino acid sequences and motifs, Bioinformatics, Volume 35, Issue 17, 1 September 2019, Pages 2974–2981,
    https://doi.org/10.1093/bioinformatics/btz035

    Note:
        - OLGA model generates sequences that correspond to IMGT junction and are used for matching as such. See the https://github.com/statbiophys/OLGA for more details.
        - Gene names are as provided in OLGA (either in default models or in the user-specified model files). For simulation, one should use gene names in the same format.

    """
    model_path: Path
    default_model_name: str
    chain: Chain = None
    use_only_productive: bool = True
    _olga_model: InternalOlgaModel = None

    DEFAULT_MODEL_FOLDER_MAP = {
        "humanTRA": "human_T_alpha", "humanTRB": "human_T_beta",
        "humanIGH": "human_B_heavy", "humanIGK": "human_B_kappa", "humanIGL": "human_B_lambda",
        "mouseTRB": "mouse_T_beta", "mouseTRA": "mouse_T_alpha"
    }
    MODEL_FILENAMES = {'marginals': 'model_marginals.txt', 'params': 'model_params.txt', 'v_gene_anchor': 'V_gene_CDR3_anchors.csv',
                       'j_gene_anchor': 'J_gene_CDR3_anchors.csv'}
    OUTPUT_COLUMNS = ["sequence", 'sequence_aa', 'v_call', 'j_call', 'region_type', "frame_type", "p_gen"]

    @classmethod
    def build_object(cls, **kwargs):

        location = OLGA.__name__

        ParameterValidator.assert_keys_present(list(kwargs.keys()), ['model_path', 'default_model_name', 'use_only_productive'], location,
                                               'OLGA generative model')
        ParameterValidator.assert_type_and_value(kwargs['use_only_productive'], bool, location, 'use_only_productive')

        if kwargs['model_path']:
            assert 'chain' in kwargs, f"{OLGA.__name__}: chain not defined."
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

    @property
    def is_vdj(self):
        return self.chain in [Chain.BETA, Chain.HEAVY]

    def load_model(self, model_path: Path = None) -> InternalOlgaModel:
        model_path = self.model_path if model_path is None else model_path
        olga_gen_model = load_model.GenerativeModelVDJ() if self.is_vdj else load_model.GenerativeModelVJ()
        olga_gen_model.load_and_process_igor_model(str(model_path / OLGA.MODEL_FILENAMES['marginals']))

        genomic_data = load_model.GenomicDataVDJ() if self.is_vdj else load_model.GenomicDataVJ()
        genomic_data.load_igor_genomic_data(params_file_name=str(model_path / OLGA.MODEL_FILENAMES['params']),
                                            V_anchor_pos_file=str(model_path / OLGA.MODEL_FILENAMES['v_gene_anchor']),
                                            J_anchor_pos_file=str(model_path / OLGA.MODEL_FILENAMES['j_gene_anchor']))

        v_gene_mapping = [gene[0] for gene in genomic_data.genV]
        j_gene_mapping = [gene[0] for gene in genomic_data.genJ]

        sequence_gen_model = SequenceGenerationVDJ(olga_gen_model, genomic_data) if self.is_vdj \
            else SequenceGenerationVJ(olga_gen_model, genomic_data)

        return InternalOlgaModel(sequence_gen_model=sequence_gen_model, v_gene_mapping=v_gene_mapping, j_gene_mapping=j_gene_mapping,
                                 genomic_data=genomic_data, olga_gen_model=olga_gen_model)

    def generate_sequences(self, count: int, seed: int = 1, path: Path = None, sequence_type: SequenceType = SequenceType.AMINO_ACID) -> Path:

        if not self._olga_model:
            self._olga_model = self.load_model()

        self._generate_productive_sequences(count, path, seed, self._olga_model)

        return path

    def _generate_productive_sequences(self, count: int, path: Path, seed: int, olga_model: InternalOlgaModel, **kwargs):
        sequences = pd.DataFrame(index=np.arange(count), columns=OLGA.OUTPUT_COLUMNS)

        for i in range(count):
            seq_row = olga_model.sequence_gen_model.gen_rnd_prod_CDR3()
            sequences.loc[i] = (seq_row[0], seq_row[1], olga_model.v_gene_mapping[seq_row[2]], olga_model.j_gene_mapping[seq_row[3]],
                                RegionType.IMGT_JUNCTION.name, SequenceFrameType.IN.name, -1.)

        sequences.to_csv(path, index=False, sep='\t')
        return path

    def compute_p_gens(self, sequences: BNPDataClass, sequence_type: SequenceType) -> list:

        cls = GenerationProbabilityVDJ if self.is_vdj else GenerationProbabilityVJ
        p_gen_model = cls(generative_model=self._olga_model.olga_gen_model, genomic_data=self._olga_model.genomic_data)
        p_gen_func = p_gen_model.compute_nt_CDR3_pgen if sequence_type == SequenceType.NUCLEOTIDE else p_gen_model.compute_aa_CDR3_pgen
        seq_field = 'sequence' if sequence_type == SequenceType.NUCLEOTIDE else 'sequence_aa'

        return [p_gen_func(getattr(seq, seq_field).to_string(), seq.v_call.to_string(), seq.j_call.to_string(), False) for seq in sequences]

    def can_compute_p_gens(self) -> bool:
        return True

    def can_generate_from_skewed_gene_models(self) -> bool:
        return True

    def generate_from_skewed_gene_models(self, v_genes: list, j_genes: list, seed: int, path: Path, sequence_type: SequenceType, batch_size: int):
        if len(v_genes) > 0 or len(j_genes) > 0:

            skewed_model_path = PathBuilder.build(path.parent / "skewed_model/")
            skewed_seqs_path = PathBuilder.build(path.parent) / f"tmp_skewed_model_seqs_{seed}.tsv"
            make_skewed_model_files(v_genes, j_genes, self.model_path, skewed_model_path)
            skewed_model = self.load_model(skewed_model_path)

            self._generate_productive_sequences(count=batch_size, path=skewed_seqs_path, seed=seed, olga_model=skewed_model)

            if skewed_seqs_path.is_file():
                pd.read_csv(skewed_seqs_path, sep='\t').to_csv(path, mode='a', sep='\t', index=False, header=False)
                os.remove(skewed_seqs_path)

    def _import_olga_sequences(self, sequence_type: SequenceType, path: Path):
        import_empty_nt_sequences = False if sequence_type == SequenceType.NUCLEOTIDE else True
        default_params = DefaultParamsLoader.load('datasets', 'olga')
        params = DatasetImportParams.build_object(**{**default_params, **{'path': path, "import_empty_nt_sequences": import_empty_nt_sequences}})
        sequences = ImportHelper.import_items(OLGAImport, path, params)
        return sequences
