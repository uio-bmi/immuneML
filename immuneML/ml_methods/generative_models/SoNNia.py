import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from olga import load_model

from immuneML.data_model.SequenceParams import RegionType, Chain
from immuneML.data_model.bnp_util import write_yaml, read_yaml, get_sequence_field_name
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
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

    **Specification arguments:**

    - locus (str): The locus of the receptor chain.

    - batch_size (int): number of sequences to use in each batch

    - epochs (int): number of epochs to train the model

    - deep (bool): whether to use a deep model

    - include_joint_genes (bool)

    - n_gen_seqs (int)

    - custom_model_path (str): path for the custom OLGA model if used

    - default_model_name (str): name of the default OLGA model if used

    - seed (int): random seed for the model or None


     **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ml_methods:
                my_sonnia_model:
                    SoNNia:
                        batch_size: 1e4
                        epochs: 5
                        default_model_name: humanTRB
                        deep: False
                        include_joint_genes: True
                        n_gen_seqs: 100

    """

    @classmethod
    def load_model(cls, path: Path):
        from sonnia.sonnia import SoNNia as InternalSoNNia

        assert path.exists(), f"{cls.__name__}: {path} does not exist."

        model_overview_file = path / 'model_overview.yaml'

        for file in [model_overview_file]:
            assert file.exists(), f"{cls.__name__}: {file} is not a file."

        model_overview = read_yaml(model_overview_file)
        sonnia = SoNNia(**{k: v for k, v in model_overview.items() if k != 'type'})
        with open(path / 'model.json', 'r') as json_file:
            model_data = json.load(json_file)

        sonnia._model = InternalSoNNia(custom_pgen_model=sonnia._model_path, seed=sonnia.seed,
                                       vj=sonnia.locus in [Chain.ALPHA, Chain.KAPPA, Chain.LIGHT],
                                       include_joint_genes=sonnia.include_joint_genes,
                                       include_indep_genes=not sonnia.include_joint_genes)

        sonnia._model.model.set_weights([np.array(w) for w in model_data['model_weights']])

        return sonnia

    def __init__(self, locus=None, batch_size: int = None, epochs: int = None, deep: bool = False, name: str = None,
                 default_model_name: str = None, n_gen_seqs: int = None, include_joint_genes: bool = True,
                 custom_model_path: str = None, seed: int = None):

        if locus is not None:
            super().__init__(Chain.get_chain(str(locus)), region_type=RegionType.IMGT_JUNCTION, seed=seed)
        elif default_model_name is not None:
            super().__init__(locus=Chain.get_chain(default_model_name[-3:]), region_type=RegionType.IMGT_JUNCTION,
                             seed=seed)
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
        from sonnia.sonnia import SoNNia as InternalSoNNia

        print_log(f"{SoNNia.__name__}: fitting a selection model...", True)

        data = dataset.data.topandas()[['junction_aa', 'v_call', 'j_call']]
        data_seqs = data.to_records(index=False).tolist()

        self._model = InternalSoNNia(data_seqs=data_seqs,
                                     gen_seqs=[],
                                     custom_pgen_model=self._model_path,
                                     vj=self.locus in [Chain.ALPHA, Chain.KAPPA, Chain.LIGHT],
                                     include_joint_genes=self.include_joint_genes,
                                     include_indep_genes=not self.include_joint_genes)

        self._model.add_generated_seqs(num_gen_seqs=self.n_gen_seqs, custom_model_folder=self._model_path)

        self._model.infer_selection(epochs=self.epochs, batch_size=self.batch_size, verbose=1)

        print_log(f"{SoNNia.__name__}: selection model fitted.", True)

    def is_same(self, model) -> bool:
        raise NotImplementedError

    def generate_sequences(self, count: int, seed: int, path: Path, sequence_type: SequenceType, compute_p_gen: bool):
        from sonia.sequence_generation import SequenceGeneration

        gen_model = SequenceGeneration(self._model)
        sequences = gen_model.generate_sequences_post(count)

        df = pd.DataFrame({
            get_sequence_field_name(self.region_type, SequenceType.AMINO_ACID): [seq[0] for seq in sequences],
            get_sequence_field_name(self.region_type, SequenceType.NUCLEOTIDE): [seq[3] for seq in sequences],
            'v_call': [seq[1] for seq in sequences],
            'j_call': [seq[2] for seq in sequences],
            'gen_model_name': [self.name] * count,
            'locus': [self.locus.to_string()] * count
        })

        return SequenceDataset.build_from_partial_df(df, path=PathBuilder.build(path), name='SoNNiaDataset',
                                                     labels={'gen_model_name': [self.name]},
                                                     type_dict={'gen_model_name': str})

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

        write_yaml(path / 'model/model_overview.yaml', {'type': 'SoNNia', 'locus': self.locus.name,
                                                        **{k: v for k, v in vars(self).items()
                                                           if
                                                           k not in ['_model', 'locus', '_model_path', 'region_type']}})
        attributes_to_save = ['data_seqs', 'gen_seqs', 'log']
        self._model.save_model(path / 'model', attributes_to_save)

        model_json = self._model.model.to_json()
        model_weights = [w.tolist() for w in self._model.model.get_weights()]
        model_data = {'model_config': model_json, 'model_weights': model_weights}
        with open(path / 'model' / 'model.json', 'w') as json_file:
            json.dump(model_data, json_file)

        return Path(shutil.make_archive(str(path / 'trained_model'), "zip", str(path / 'model'))).absolute()
