from dataclasses import dataclass
from typing import List

from immuneML.data_model.SequenceParams import RegionType, Chain
from immuneML.util.CompAIRRHelper import CompAIRRHelper
from immuneML.util.ParameterValidator import ParameterValidator


@dataclass
class SequenceSimilarityConfig:
    """
    How sequences are compared when splitting a dataset by sequence similarity (group_k_fold_config/similarity in
    SplitConfig) or when measuring the overlap between training and test sequences (sequence_overlap in SplitDataset).
    The amino acid sequences are compared with CompAIRR, ignoring genes and counts. This is defined for sequence and
    receptor datasets; for repertoire datasets, group by a metadata field (e.g. subject) instead.

    - max_edit_distance (int): the largest number of differences at which two sequences count as similar, between 0
      and 2 (between 1 and 2 when grouping by similarity).

    - indels (bool): whether one of the differences at distance 1 may be an insertion or deletion. CompAIRR allows
      indels only at distance 1, so a distance of 2 always means two substitutions. By default, False.

    - region_type (str): which part of the sequence to compare, e.g. IMGT_CDR3 (the default) or IMGT_JUNCTION.

    - chains (list): for receptor datasets only, which chains to compare, e.g. [TRB]. Two receptors are within a
      distance only if every compared chain is, so the more chains are compared, the fewer receptors count as similar.
      By default, all chains of the receptors are compared.

    - compairr_path (str): path to the CompAIRR executable. If not given, it is assumed that CompAIRR can be called as
      'compairr' on the command line, or that it is at /usr/local/bin/compairr.

    - threads (int): the number of threads CompAIRR may use; this only affects speed. By default, 4.
    """

    max_edit_distance: int = 1
    indels: bool = False
    region_type: RegionType = RegionType.IMGT_CDR3
    compairr_path: str = None
    threads: int = 4
    chains: List[Chain] = None

    @staticmethod
    def parse(specs: dict, location: str, parameter_name: str, min_edit_distance: int = 0) -> 'SequenceSimilarityConfig':
        assert isinstance(specs, dict), f"{location}: {parameter_name} has to be a dictionary, got {specs} instead."
        ParameterValidator.assert_keys(specs.keys(), ['max_edit_distance', 'indels', 'region_type', 'compairr_path',
                                                      'threads', 'chains'], location, parameter_name, exclusive=False)
        ParameterValidator.assert_keys_present(specs.keys(), ['max_edit_distance'], location, parameter_name)

        ParameterValidator.assert_type_and_value(specs['max_edit_distance'], int, location,
                                                 f'{parameter_name}/max_edit_distance', min_inclusive=min_edit_distance,
                                                 max_inclusive=2)
        indels = specs.get('indels', False)
        ParameterValidator.assert_type_and_value(indels, bool, location, f'{parameter_name}/indels')
        assert not indels or specs['max_edit_distance'] >= 1, \
            f"{location}: {parameter_name}/indels has no effect when max_edit_distance is 0."

        region_type = specs.get('region_type', RegionType.IMGT_CDR3.name)
        ParameterValidator.assert_region_type({'region_type': region_type}, location)
        threads = specs.get('threads', 4)
        ParameterValidator.assert_type_and_value(threads, int, location, f'{parameter_name}/threads', min_inclusive=1)

        chains = None
        if specs.get('chains') is not None:
            ParameterValidator.assert_type_and_value(specs['chains'], list, location, f'{parameter_name}/chains')
            chains = [Chain.get_chain(str(chain)) for chain in specs['chains']]
            assert len(chains) > 0 and all(chain is not None for chain in chains), \
                (f"{location}: {parameter_name}/chains has to be a non-empty list of chains (e.g. TRA, TRB, IGH), "
                 f"got {specs['chains']}.")

        if specs['max_edit_distance'] > 0:
            CompAIRRHelper.determine_compairr_path(specs.get('compairr_path'))

        return SequenceSimilarityConfig(max_edit_distance=specs['max_edit_distance'], indels=indels,
                                        region_type=RegionType[region_type.upper()],
                                        compairr_path=specs.get('compairr_path'), threads=threads, chains=chains)

    def describe(self) -> str:
        indels = ", one of them may be an indel" if self.indels and self.max_edit_distance >= 1 else ""
        chains = f" ({', '.join(chain.value for chain in self.chains)})" if self.chains else ""
        return f"{self.region_type.name} sequences{chains} within {self.max_edit_distance} differences{indels}"
