import logging
import shutil
from pathlib import Path

import yaml

from immuneML.api.galaxy.GalaxyTool import GalaxyTool
from immuneML.api.galaxy.Util import Util
from immuneML.workflows.instructions.SimulationInstruction import SimulationInstruction


class GalaxySimulationTool(GalaxyTool):

    """
    GalaxySimulationTool is an alternative to running immuneML with the simulation instruction directly. It accepts a YAML specification file and a
    path to the output directory. It implants the signals in the dataset that was provided either as an existing dataset with a set of files or in
    the random dataset as described in the specification file.

    This tool is meant to be used as an endpoint for Galaxy tool that will create a Galaxy collection out of a dataset in immuneML format that can
    be readily used by other immuneML-based Galaxy tools.

    The specification supplied for this tool is identical to immuneML specification, except that it can include only one instruction which has to
    be of type 'Simulation':

    .. code-block: yaml

        definitions:
            datasets:
                my_synthetic_dataset:
                    format: RandomRepertoireDataset
                    params:
                        repertoire_count: 100
                        labels: {}
            motifs:
                my_simple_motif: # a simple motif without gaps or hamming distance
                  seed: AAA
                  instantiation: GappedKmer

                my_complex_motif: # complex motif containing a gap + hamming distance
                    seed: AA/A  # ‘/’ denotes gap position if present, if not, there’s no gap
                    instantiation:
                        GappedKmer:
                            min_gap: 1
                            max_gap: 2
                            hamming_distance_probabilities: # probabilities for each number of
                                0: 0.7                    # modification to the seed
                                1: 0.3
                            position_weights: # probabilities for modification per position
                                0: 1
                                1: 0 # note that index 2, the position of the gap,
                                3: 0 # is excluded from position_weights
                            alphabet_weights: # probabilities for using each amino acid in
                                A: 0.2      # a hamming distance modification
                                C: 0.2
                                D: 0.4
                                E: 0.2

            signals:
                my_signal:
                    motifs:
                    - my_simple_motif
                    - my_complex_motif
                    implanting: HealthySequence
                    sequence_position_weights:
                        109: 1
                        110: 2
                        111: 5
                        112: 1
            simulations:
                my_simulation:
                    my_implanting:
                        signals:
                        - my_signal
                        dataset_implanting_rate: 0.5
                        repertoire_implanting_rate: 0.25
        instructions:
            my_simulation_instruction: # user-defined name of the instruction
                type: Simulation # which instruction to execute
                dataset: my_dataset # which dataset to use for implanting the signals
                simulation: my_simulation # how to implanting the signals - definition of the simulation
                number_of_processes: 4 # how many parallel processes to use during execution
                export_formats: [AIRR] # in which formats to export the dataset, Pickle format will be added automatically
        output: # the output format
            format: HTML

    """

    def __init__(self, specification_path: Path, result_path: Path, **kwargs):
        Util.check_parameters(specification_path, result_path, kwargs, GalaxySimulationTool.__name__)
        super().__init__(specification_path, result_path, **kwargs)

    def _run(self):
        self.prepare_specs()

        Util.run_tool(self.yaml_path, self.result_path)

        dataset_location = list(self.result_path.glob("*/exported_dataset/*/"))[0]
        shutil.copytree(dataset_location, self.result_path / 'result/')

        logging.info(f"{GalaxySimulationTool.__name__}: immuneML has finished and the signals were implanted in the dataset.")

    def prepare_specs(self):
        with self.yaml_path.open("r") as file:
            specs = yaml.safe_load(file)

        instruction_name = Util.check_instruction_type(specs, GalaxySimulationTool.__name__, SimulationInstruction.__name__[:-11])
        Util.check_export_format(specs, GalaxySimulationTool.__name__, instruction_name)

        Util.check_paths(specs, "GalaxySimulationTool")
        Util.update_result_paths(specs, self.result_path, self.yaml_path)
