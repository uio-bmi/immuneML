import os
import shutil
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
from immuneML.data_model.bnp_util import read_yaml, write_yaml
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.ml_methods.generative_models.InternalOlgaModel import InternalOlgaModel
from immuneML.simulation.util.igor_helper import make_skewed_model_files
from immuneML.util.ImportHelper import ImportHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


@dataclass
class OLGA(GenerativeModel):
    """
    This is a wrapper for the OLGA package as described by Sethna et al. 2019 (OLGA package on PyPI or GitHub: https://github.com/statbiophys/OLGA).

    Reference:

    Zachary Sethna, Yuval Elhanati, Curtis G Callan, Jr, Aleksandra M Walczak, Thierry Mora, OLGA: fast computation of
    generation probabilities of B- and T-cell receptor amino acid sequences and motifs, Bioinformatics, Volume 35,
    Issue 17, 1 September 2019, Pages 2974–2981, https://doi.org/10.1093/bioinformatics/btz035

    Note:

    - OLGA generates sequences that correspond to IMGT junction and are used for matching as such. See the
      https://github.com/statbiophys/OLGA for more details.

    - Gene names are as provided in OLGA (either in default models or in the user-specified model files). For
      simulation, one should use gene names in the same format.

    .. note::

        While this is a generative model, in version 3.0.0a1, it cannot be used in combination with TrainGenModel or
        ApplyGenModel instruction. If you want to use OLGA for sequence simulation, see :ref:`How to simulate antigen
        or disease-associated signals in AIRR datasets`.
`

    Specification arguments:

    - model_path (str): if not default model, this parameter should point to a folder where the four OLGA/IGOR format
      files are stored (could also be inferred from some experimental data)

    - default_model_name (str): if not using custom models, one of the OLGA default models could be specified here;
      the value should be the same as it would be passed to command line in OLGA: e.g., humanTRB, human IGH

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        generative_model:
            type: OLGA
            model_path: None
            default_model_name: humanTRB

    """

    model_path: Path = None
    default_model_name: str = None
    chain: Chain = None
    region_type: RegionType = RegionType.IMGT_JUNCTION
    _olga_model: InternalOlgaModel = None

    DEFAULT_MODEL_FOLDER_MAP = {
        "humanTRA": "human_T_alpha", "humanTRB": "human_T_beta",
        "humanIGH": "human_B_heavy", "humanIGK": "human_B_kappa", "humanIGL": "human_B_lambda",
        "mouseTRB": "mouse_T_beta", "mouseTRA": "mouse_T_alpha"
    }
    MODEL_FILENAMES = {'marginals': 'model_marginals.txt', 'params': 'model_params.txt',
                       'v_gene_anchor': 'V_gene_CDR3_anchors.csv',
                       'j_gene_anchor': 'J_gene_CDR3_anchors.csv'}
    OUTPUT_COLUMNS = ["sequence", 'sequence_aa', 'v_call', 'j_call', 'region_type', "frame_type", "p_gen",
                      "from_default_model", 'duplicate_count', 'chain']

    @classmethod
    def build_object(cls, **kwargs):

        location = OLGA.__name__

        ParameterValidator.assert_keys(list(kwargs.keys()), ['model_path', 'default_model_name', 'chain'], location,
                                       'OLGA generative model', exclusive=False)

        if 'model_path' in kwargs and kwargs['model_path']:
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
            ParameterValidator.assert_in_valid_list(kwargs['default_model_name'],
                                                    list(OLGA.DEFAULT_MODEL_FOLDER_MAP.keys()), location,
                                                    'default_model_name')
            chain = Chain.get_chain(kwargs['default_model_name'][-3:])
            kwargs['model_path'] = Path(
                load_model.__file__).parent / f"default_models/{OLGA.DEFAULT_MODEL_FOLDER_MAP[kwargs['default_model_name']]}"

        return OLGA(**{**kwargs, **{'chain': chain}})

    @classmethod
    def load_model(cls, path: Path):
        assert path.exists(), f"{cls.__name__}: {path} does not exist."

        model_overview_file = path / 'model_overview.yaml'

        for file in [model_overview_file]:
            assert file.exists(), f"{cls.__name__}: {file} is not a file."

        model_overview = read_yaml(model_overview_file)
        olga = OLGA(**{k: v for k, v in model_overview.items() if k != 'type'})
        olga._olga_model = olga.load_internal_model(path)
        return olga

    @property
    def is_vdj(self):
        return self.chain in [Chain.BETA, Chain.HEAVY]

    def fit(self, data, path: Path = None):
        raise NotImplementedError("Fitting an OLGA model is currently not supported by immuneML.")

    def load_internal_model(self, model_path: Path = None) -> InternalOlgaModel:
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

        return InternalOlgaModel(sequence_gen_model=sequence_gen_model, v_gene_mapping=v_gene_mapping,
                                 j_gene_mapping=j_gene_mapping,
                                 genomic_data=genomic_data, olga_gen_model=olga_gen_model)

    def generate_sequences(self, count: int, seed: int, path: Path, sequence_type: SequenceType,
                           compute_p_gen: bool) -> Path:

        if not self._olga_model:
            self._olga_model = self.load_internal_model()

        self._generate_productive_sequences(count, path, seed, self._olga_model, compute_p_gen, sequence_type)

        return path

    def _generate_productive_sequences(self, count: int, path: Path, seed: int, olga_model: InternalOlgaModel,
                                       compute_p_gen: bool,
                                       sequence_type: SequenceType, **kwargs):
        sequences = pd.DataFrame(index=np.arange(count), columns=OLGA.OUTPUT_COLUMNS)

        for i in range(count):
            seq_row = olga_model.sequence_gen_model.gen_rnd_prod_CDR3()

            p_gen = self.compute_p_gen(
                {'sequence': seq_row[0], 'sequence_aa': seq_row[1], 'v_call': olga_model.v_gene_mapping[seq_row[2]],
                 'j_call': olga_model.j_gene_mapping[seq_row[3]]}, sequence_type) if compute_p_gen else -1.

            sequences.loc[i] = (
            seq_row[0], seq_row[1], olga_model.v_gene_mapping[seq_row[2]], olga_model.j_gene_mapping[seq_row[3]],
            RegionType.IMGT_JUNCTION.name, SequenceFrameType.IN.value, p_gen, int(olga_model == self._olga_model),
            -1, self.chain.value)

        sequences.to_csv(path, index=False, sep='\t')
        return path

    def compute_p_gen(self, sequence: dict, sequence_type: SequenceType, sequence_field: str = None) -> float:
        cls = GenerationProbabilityVDJ if self.is_vdj else GenerationProbabilityVJ
        p_gen_model = cls(generative_model=self._olga_model.olga_gen_model, genomic_data=self._olga_model.genomic_data)
        if sequence_type == SequenceType.NUCLEOTIDE:
            seq_field_name = 'sequence' if sequence_field is None else sequence_field
            return p_gen_model.compute_nt_CDR3_pgen(sequence[seq_field_name], sequence['v_call'], sequence['j_call'])
        else:
            seq_field_name = 'sequence_aa' if sequence_field is None else sequence_field
            return p_gen_model.compute_aa_CDR3_pgen(sequence[seq_field_name], sequence['v_call'], sequence['j_call'])

    def compute_p_gens(self, sequences: BNPDataClass, sequence_type: SequenceType, sequence_field: str = None) -> list:

        cls = GenerationProbabilityVDJ if self.is_vdj else GenerationProbabilityVJ
        p_gen_model = cls(generative_model=self._olga_model.olga_gen_model, genomic_data=self._olga_model.genomic_data)
        p_gen_func = p_gen_model.compute_nt_CDR3_pgen if sequence_type == SequenceType.NUCLEOTIDE else p_gen_model.compute_aa_CDR3_pgen
        if sequence_field is None:
            seq_field = 'sequence' if sequence_type == SequenceType.NUCLEOTIDE else 'sequence_aa'
        else:
            seq_field = sequence_field

        return [p_gen_func(getattr(seq, seq_field).to_string(), seq.v_call.to_string(), seq.j_call.to_string(), False)
                for seq in sequences]

    def can_compute_p_gens(self) -> bool:
        return True

    def can_generate_from_skewed_gene_models(self) -> bool:
        return True

    def generate_from_skewed_gene_models(self, v_genes: list, j_genes: list, seed: int, path: Path,
                                         sequence_type: SequenceType, batch_size: int,
                                         compute_p_gen: bool):

        if not self._olga_model:
            self._olga_model = self.load_internal_model()

        if len(v_genes) > 0 or len(j_genes) > 0:

            skewed_model_path = PathBuilder.build(path.parent / "skewed_model/")
            skewed_seqs_path = PathBuilder.build(path.parent) / f"tmp_skewed_model_seqs_{seed}.tsv"
            make_skewed_model_files(v_genes, j_genes, self.model_path, skewed_model_path)
            skewed_model = self.load_internal_model(skewed_model_path)

            self._generate_productive_sequences(count=batch_size, path=skewed_seqs_path, seed=seed,
                                                olga_model=skewed_model,
                                                compute_p_gen=compute_p_gen, sequence_type=sequence_type)

            if skewed_seqs_path.is_file():
                pd.read_csv(skewed_seqs_path, sep='\t').to_csv(path, mode='a' if path.is_file() else 'w', sep='\t',
                                                               index=False,
                                                               header=not path.is_file())
                os.remove(skewed_seqs_path)

    def _import_olga_sequences(self, sequence_type: SequenceType, path: Path):
        import_empty_nt_sequences = False if sequence_type == SequenceType.NUCLEOTIDE else True
        default_params = DefaultParamsLoader.load('datasets', 'olga')
        params = DatasetImportParams.build_object(
            **{**default_params, **{'path': path, "import_empty_nt_sequences": import_empty_nt_sequences}})
        sequences = ImportHelper.import_items(OLGAImport, path, params)
        return sequences

    def is_same(self, model) -> bool:
        return type(model) == type(self) and self.chain == model.chain and self.model_path == model.model_path and \
            self.default_model_name == model.default_model_name

    def save_model(self, path: Path) -> Path:
        PathBuilder.build(path / 'model')

        write_yaml(path / 'model_overview.yaml',
                   {'type': 'OLGA', 'default_model_name': self.default_model_name, 'chain': self.chain.name,
                    'region_type': self.region_type.name})

        shutil.copytree(str(self.model_path), str(path / 'model'), dirs_exist_ok=True)
        return Path(shutil.make_archive(str(path / 'trained_model'), "zip", str(path / 'model'))).absolute()

