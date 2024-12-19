import json
import shutil
from pathlib import Path

import numpy as np
from olga import load_model

from immuneML.data_model.bnp_util import write_yaml, read_yaml
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.data_model.SequenceParams import RegionType, Chain
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.ml_methods.generative_models.OLGA import OLGA
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder


class SONIA(GenerativeModel):
    """
    Sonia models the selection process of T and B cell receptor repertoires. It is based on the Sonia Python package
    and uses the Left+Right model. It supports SequenceDataset as input, but not RepertoireDataset.

    Original publication:
    Sethna Z, Isacchini G, Dupic T, Mora T, Walczak AM, et al. (2020) Population variability in the generation and
    selection of T-cell repertoires. PLOS Computational Biology 16(12): e1008394.
    https://doi.org/10.1371/journal.pcbi.1008394

    **Specification arguments:**

    - locus (str)

    - batch_size (int)

    - epochs (int)

    - include_joint_genes (bool)

    - n_gen_seqs (int)

    - custom_model_path (str)

    - default_model_name (str)

        **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ml_methods:
                my_sonia_model:
                    SONIA:
                        ...

    """

    @classmethod
    def load_model(cls, path: Path):
        from sonia.sonia_leftpos_rightpos import SoniaLeftposRightpos as InternalSONIA

        assert path.exists(), f"{cls.__name__}: {path} does not exist."

        model_overview_file = path / 'model_overview.yaml'

        for file in [model_overview_file]:
            assert file.exists(), f"{cls.__name__}: {file} is not a file."

        model_overview = read_yaml(model_overview_file)
        sonia = SONIA(**{k: v for k, v in model_overview.items() if k != 'type'})
        with open(path / 'model.json', 'r') as json_file:
            model_data = json.load(json_file)

        sonia._model = InternalSONIA(custom_pgen_model=sonia._model_path,
                                     vj=sonia.locus in [Chain.ALPHA, Chain.KAPPA, Chain.LIGHT],
                                     include_joint_genes=sonia.include_joint_genes,
                                     include_indep_genes=not sonia.include_joint_genes)

        sonia._model.model.set_weights([np.array(w) for w in model_data['model_weights']])

        return sonia

    def __init__(self, locus=None, batch_size: int = None, epochs: int = None, name: str = None,
                 default_model_name: str = None, n_gen_seqs: int = None, include_joint_genes: bool = True,
                 custom_model_path: str = None):

        if locus is not None:
            super().__init__(locus, region_type=RegionType.IMGT_JUNCTION)
        elif default_model_name is not None:
            super().__init__(locus=Chain.get_chain(default_model_name[-3:]), region_type=RegionType.IMGT_JUNCTION)
        self.epochs = epochs
        self.batch_size = int(batch_size)
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
        from sonia.sonia_leftpos_rightpos import SoniaLeftposRightpos as InternalSONIA

        print_log(f"{SONIA.__name__}: fitting a selection model...", True)

        data = dataset.data.topandas()[['junction_aa', 'v_call', 'j_call']]
        data_seqs = data.to_records(index=False).tolist()

        self._model = InternalSONIA(data_seqs=data_seqs,
                                    gen_seqs=[],
                                    chain_type=self.default_model_name,
                                    custom_pgen_model=self._model_path,
                                    vj=self.locus in [Chain.ALPHA, Chain.KAPPA, Chain.LIGHT],
                                    include_joint_genes=self.include_joint_genes,
                                    include_indep_genes=not self.include_joint_genes
                                    )

        self._model.add_generated_seqs(num_gen_seqs=self.n_gen_seqs, custom_model_folder=self._model_path)

        self._model.infer_selection(epochs=self.epochs, batch_size=self.batch_size, verbose=1)

        print_log(f"{SONIA.__name__}: selection model fitted.", True)

    def generate_sequences(self, count: int, seed: int, path: Path, sequence_type: SequenceType, compute_p_gen: bool):
        from sonia.sequence_generation import SequenceGeneration

        gen_model = SequenceGeneration(self._model)
        sequences = gen_model.generate_sequences_post(count)

        return SequenceDataset.build_from_objects(sequences=[ReceptorSequence(sequence_aa=seq[0], sequence=seq[3],
                                                                              v_call=seq[1], j_call=seq[2],
                                                                              metadata={'gen_model_name': self.name if self.name else "Sonia"})
                                                                                for seq in sequences],
                                                  region_type=RegionType.IMGT_JUNCTION,
                                                  path=PathBuilder.build(path), name='SoniaDataset',
                                                  labels={'gen_model_name': [self.name]})

    def can_compute_p_gens(self) -> bool:
        return False

    def can_generate_from_skewed_gene_models(self) -> bool:
        return False

    def save_model(self, path: Path) -> Path:
        PathBuilder.build(path / 'model')

        write_yaml(path / 'model/model_overview.yaml', {'type': 'SONIA', 'locus': self.locus.name,
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
