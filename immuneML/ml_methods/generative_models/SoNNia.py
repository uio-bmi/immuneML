import shutil
from pathlib import Path

import numpy as np
from olga import load_model
from sonia.sequence_generation import SequenceGeneration
from sonnia.sonnia import SoNNia as InternalSoNNia

from immuneML.data_model.bnp_util import write_yaml, read_yaml
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.ml_methods.generative_models.OLGA import OLGA
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder


class SoNNia(GenerativeModel):
    """
    SoNNia models the selection process of T and B cell receptor repertoires. It is based on the SoNNia Python package.
    It supports SequenceDataset as input, but not RepertoireDataset.

    Original publication:
    Isacchini, G., Walczak, A. M., Mora, T., & Nourmohammad, A. (2021). Deep generative selection models of T and B
    cell receptor repertoires with soNNia. Proceedings of the National Academy of Sciences, 118(14), e2023141118.
    https://doi.org/10.1073/pnas.2023141118

    Specification arguments:

    - chain (str)

    - batch_size (int)

    - epochs (int)

    - deep (bool)

    - include_joint_genes (bool)

    - n_gen_seqs (int)

    - custom_model_path (str)

    - default_model_name (str)

    """

    @classmethod
    def load_model(cls, path: Path):
        assert path.exists(), f"{cls.__name__}: {path} does not exist."

        model_overview_file = path / 'model_overview.yaml'

        for file in [model_overview_file]:
            assert file.exists(), f"{cls.__name__}: {file} is not a file."

        model_overview = read_yaml(model_overview_file)
        sonnia = SoNNia(**{k: v for k, v in model_overview.items() if k != 'type'})
        sonnia._model = InternalSoNNia(load_dir=path)
        return sonnia

    def __init__(self, chain=None, batch_size: int = None, epochs: int = None, deep: bool = False, name: str = None,
                 default_model_name: str = None, n_gen_seqs: int = None, include_joint_genes: bool = True,
                 custom_model_path: str = None):
        if chain is not None:
            super().__init__(chain)
        elif default_model_name is not None:
            super().__init__(chain=Chain.get_chain(default_model_name[-3:]))
        self.epochs = epochs
        self.batch_size = int(batch_size)
        self.deep = deep
        self.include_joint_genes = include_joint_genes
        self.n_gen_seqs = n_gen_seqs
        self._model = None
        self.name = name
        self.default_model_name = default_model_name
        if custom_model_path is None or custom_model_path == '':
            self._model_path = Path(
                load_model.__file__).parent / f"default_models/{OLGA.DEFAULT_MODEL_FOLDER_MAP[self.default_model_name]}"
        else:
            self._model_path = custom_model_path

    def fit(self, dataset: Dataset, path: Path = None):
        print_log(f"{SoNNia.__name__}: fitting a selection model...", True)

        data = dataset.get_attributes(['sequence_aa', 'v_call', 'j_call'], as_list=True)
        data_seqs = [[data['sequence_aa'][i], data['v_call'][i], data['j_call'][i]]
                     for i in range(len(data['sequence_aa']))]

        self._model = InternalSoNNia(data_seqs=data_seqs,
                                     gen_seqs=[],
                                     custom_pgen_model=self._model_path,
                                     vj=self.chain in [Chain.ALPHA, Chain.KAPPA, Chain.LIGHT],
                                     include_joint_genes=self.include_joint_genes,
                                     include_indep_genes=not self.include_joint_genes)

        self._model.add_generated_seqs(num_gen_seqs=self.n_gen_seqs, custom_model_folder=self._model_path)

        self._model.infer_selection(epochs=self.epochs, batch_size=self.batch_size, verbose=1)

        print_log(f"{SoNNia.__name__}: selection model fitted.", True)

    def is_same(self, model) -> bool:
        raise NotImplementedError

    def generate_sequences(self, count: int, seed: int, path: Path, sequence_type: SequenceType, compute_p_gen: bool):
        gen_model = SequenceGeneration(self._model)
        sequences = gen_model.generate_sequences_post(count)
        return SequenceDataset.build_from_objects(sequences=[ReceptorSequence(sequence_aa=seq[0], sequence=seq[3],
                                                                              metadata=SequenceMetadata(
                                                                                  v_call=seq[1],
                                                                                  j_call=seq[2],
                                                                                  region_type=RegionType.IMGT_JUNCTION.name))
                                                             for seq in sequences],
                                                  file_size=len(sequences), path=PathBuilder.build(path),
                                                  name='SoNNiaDataset')

    def compute_p_gens(self, sequences, sequence_type: SequenceType) -> np.ndarray:
        raise NotImplementedError

    def compute_p_gen(self, sequence: dict, sequence_type: SequenceType) -> float:
        raise NotImplementedError

    def can_compute_p_gens(self) -> bool:
        return False

    def can_generate_from_skewed_gene_models(self) -> bool:
        return False

    def generate_from_skewed_gene_models(self, v_genes: list, j_genes: list, seed: int, path: Path,
                                         sequence_type: SequenceType, batch_size: int, compute_p_gen: bool):
        raise NotImplementedError

    def save_model(self, path: Path) -> Path:
        PathBuilder.build(path / 'model')

        write_yaml(path / 'model/model_overview.yaml', {'type': 'SoNNia', 'chain': self.chain.name,
                                                        **{k: v for k, v in vars(self).items()
                                                           if k not in ['_model', 'chain', '_model_path']}})
        self._model.save_model(path / 'model')

        return Path(shutil.make_archive(str(path / 'trained_model'), "zip", str(path / 'model'))).absolute()
