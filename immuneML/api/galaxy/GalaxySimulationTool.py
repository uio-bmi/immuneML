import logging
import shutil
from pathlib import Path

import yaml

from immuneML.api.galaxy.GalaxyTool import GalaxyTool
from immuneML.api.galaxy.Util import Util
from immuneML.workflows.instructions.ligo_simulation.LigoSimInstruction import LigoSimInstruction


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
          motifs:
            motif1:
              seed: AA
            motif2:
              seed: GG
          signals:
            signal1:
              motifs: [motif1]
            signal2:
              motifs: [motif2]
          simulations:
            sim1:
              is_repertoire: true
              paired: false
              sequence_type: amino_acid
              simulation_strategy: Implanting
              remove_seqs_with_signals: true # remove signal-specific AIRs from the background
              sim_items:
                sim_item: # group of AIRs with the same parameters
                  AIRR1:
                    immune_events:
                      ievent1: True
                      ievent1: False
                    signals: [signal1: 0.3, signal2: 0.3]
                    number_of_examples: 10
                    is_noise: False
                    receptors_in_repertoire_count: 6,
                    generative_model:
                      chain: heavy
                      default_model_name: humanIGH
                      model_path: null
                      type: OLGA
                  AIRR2:
                    immune_events:
                      ievent1: False
                      ievent1: True
                    signals: [signal1: 0.5, signal2: 0.5]
                    number_of_examples: 10
                    is_noise: False
                    receptors_in_repertoire_count: 6,
                    generative_model:
                      chain: heavy
                      default_model_name: humanIGH
                      model_path: null
                      type: OLGA
          instructions:
            my_sim_inst:
              export_p_gens: false
              max_iterations: 100
              number_of_processes: 4
              sequence_batch_size: 1000
              simulation: sim1
              type: LigoSim

    """

    def __init__(self, specification_path: Path, result_path: Path, **kwargs):
        Util.check_parameters(specification_path, result_path, kwargs, GalaxySimulationTool.__name__)
        super().__init__(specification_path, result_path, **kwargs)

    def _run(self):
        self.prepare_specs()

        Util.run_tool(self.yaml_path, self.result_path)

        dataset_location = list(self.result_path.glob("*/exported_dataset/*/"))[0]
        shutil.copytree(dataset_location, self.result_path / 'result/')

        logging.info(f"{GalaxySimulationTool.__name__}: the simulation is finished.")

    def prepare_specs(self):
        with self.yaml_path.open("r") as file:
            specs = yaml.safe_load(file)

        Util.check_paths(specs, "GalaxySimulationTool")
        Util.update_result_paths(specs, self.result_path, self.yaml_path)
