import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.IO.dataset_import.OLGAImport import OLGAImport
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
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

    @classmethod
    def build_object(cls, **kwargs):

        location = OLGA.__name__

        ParameterValidator.assert_keys(kwargs.keys(), ['model_path', 'default_model_name'], location, 'OLGA generative model')

        if kwargs['model_path']:
            assert Path(kwargs['model_path']).is_dir(), \
                f"{OLGA.__name__}: the model path is not a directory. It has to be a directory and contain files with the exact names as " \
                f"described in the OLGA package documentation: https://github.com/statbiophys/OLGA."

            files = glob.glob(kwargs['model_path'] + "/*.txt")
            assert all(file in files for file in [kwargs['model_path'] + "/model_params.txt", kwargs['model_path'] + "/model_marginals.txt"]), \
                f"{OLGA.__name__}: model files are missing in the specified directory: {kwargs['model_path']}"

            ParameterValidator.assert_type_and_value(kwargs['default_model_name'], None, location, 'default_model_name', exact_value=None)
        else:
            ParameterValidator.assert_in_valid_list(kwargs['default_model_name'], ['humanTRA', 'humanTRB', 'mouseTRB', 'humanIGH'], location,
                                                    'default_model_name')

        return OLGA(model_path=kwargs['model_path'], default_model_name=kwargs['default_model_name'])

    def load_model(self):
        pass

    def generate_sequences(self, count: int, seed: int = 1, path: Path = None, sequence_type: SequenceType = SequenceType.AMINO_ACID) -> List[
        ReceptorSequence]:

        command = f"olga-generate_sequences -n {count} --seed={seed} -o {path}"
        if self.default_model_name:
            command += f" --{self.default_model_name}"
        else:
            command += f""

        os.system(command)

        import_empty_nt_sequences = False if sequence_type == SequenceType.NUCLEOTIDE else True

        default_params = DefaultParamsLoader.load('datasets', 'olga')
        params = DatasetImportParams.build_object(**{**default_params, **{'path': path, "import_empty_nt_sequences": import_empty_nt_sequences}})
        sequences = ImportHelper.import_items(OLGAImport, path, params)

        assert len(sequences) == count, f"{OLGA.__name__}: an error occurred while generating sequences. Expected {count} sequences, " \
                                        f"but got {len(sequences)} instead. See the generated sequences at {path}."

        os.remove(path)

        return sequences

    def compute_p_gens(self):
        pass

    def can_compute_p_gens(self) -> bool:
        return True
