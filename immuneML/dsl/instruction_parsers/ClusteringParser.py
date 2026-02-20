import copy
import inspect
import logging
from pathlib import Path
from typing import List

import sklearn.metrics as sklearn_metrics

from immuneML.data_model.SequenceParams import RegionType
from immuneML.dsl.instruction_parsers.LabelHelper import LabelHelper
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.hyperparameter_optimization.config.SampleConfig import SampleConfig
from immuneML.ml_methods.clustering.ClusteringMethod import ClusteringMethod
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod
from immuneML.reports.Report import Report
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.clustering.ClusteringInstruction import ClusteringInstruction
from immuneML.workflows.instructions.clustering.ClusteringState import StabilityConfig
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringSetting


class ClusteringParser:

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> ClusteringInstruction:
        valid_keys = [k for k in inspect.signature(ClusteringInstruction.__init__).parameters.keys()
                      if k not in ['result_path', 'name', 'self', 'label_config']] + ['type', 'labels']

        ParameterValidator.assert_keys(instruction.keys(), valid_keys, ClusteringParser.__name__, key)

        ParameterValidator.assert_region_type(instruction, ClusteringParser.__name__)
        ParameterValidator.assert_sequence_type(instruction, ClusteringParser.__name__)

        dataset = symbol_table.get(instruction['dataset'])
        clustering_settings = parse_clustering_settings(key, instruction, symbol_table)
        metrics = parse_metrics(key, instruction)
        label_config = parse_labels(key, instruction, dataset)
        reports = parse_reports(key, instruction, symbol_table)
        sample_config = parse_sample_config(instruction)
        stability_config = parse_stability_config(instruction)

        return ClusteringInstruction(dataset=dataset, metrics=metrics, clustering_settings=clustering_settings,
                                     name=key, label_config=label_config, reports=reports,
                                     sequence_type=SequenceType[instruction['sequence_type'].upper()],
                                     region_type=RegionType[instruction['region_type'].upper()],
                                     sample_config=sample_config, stability_config=stability_config)

def parse_stability_config(instruction) -> StabilityConfig:
    ParameterValidator.assert_type_and_value(instruction['stability_config']['split_count'], int, StabilityConfig.__name__, 'stability_config:split_count', min_inclusive=1)
    ParameterValidator.assert_type_and_value(instruction['stability_config']['random_seed'], int, StabilityConfig.__name__, 'stability_config:random_seed')
    return StabilityConfig(split_count=instruction['stability_config']['split_count'], random_seed=instruction['stability_config']['random_seed'])


def parse_sample_config(instruction) -> SampleConfig:
    split_key = 'sample_config'
    try:

        valid_keys = [k for k in inspect.signature(SampleConfig.__init__).parameters.keys() if k not in ['self']]

        ParameterValidator.assert_keys(instruction[split_key].keys(), valid_keys, ClusteringParser.__name__, split_key,
                                       exclusive=False)
        ParameterValidator.assert_type_and_value(instruction[split_key]['split_count'], int, ClusteringParser.__name__,
                                                 'split_count', min_inclusive=1)

        ParameterValidator.assert_type_and_value(instruction[split_key]['percentage'], float, ClusteringParser.__name__, 'percentage', None, None, 0, 1)
        ParameterValidator.assert_type_and_value(instruction[split_key]['random_seed'], int, ClusteringParser.__name__, 'random_seed')

        percentage = float(instruction[split_key]["percentage"])
        random_seed = int(instruction[split_key]["random_seed"])

        return SampleConfig(split_count=instruction[split_key]['split_count'], percentage=percentage, random_seed=random_seed)

    except KeyError as key_error:
        raise KeyError(
            f"{ClusteringParser.__name__}: parameter {key_error.args[0]} was not defined under {split_key}.") from key_error


def parse_labels(key, instruction, dataset) -> LabelConfiguration:
    if 'labels' in instruction and instruction['labels'] is not None:
        label_config = LabelHelper.create_label_config(instruction['labels'], dataset, key, 'labels')
    else:
        label_config = LabelConfiguration()
    return label_config


def parse_reports(key, instruction, symbol_table) -> List[Report]:
    if 'reports' not in instruction or instruction['reports'] is None:
        return []
    else:
        ParameterValidator.assert_type_and_value(instruction['reports'], list, 'ClusteringParser', f'{key}:reports')
        valid_reports = symbol_table.get_keys_by_type(SymbolType.REPORT)

        ParameterValidator.assert_all_type_and_value(instruction['reports'], str, 'ClusteringParser', f'{key}:reports')
        ParameterValidator.assert_all_in_valid_list(instruction['reports'], valid_reports, 'ClusteringParser',
                                                    f'{key}:reports')

        reports = [symbol_table.get(report_id) for report_id in instruction['reports']]
        return reports


def parse_metrics(key: str, instruction: dict) -> List[str]:
    ParameterValidator.assert_type_and_value(instruction['metrics'], list, 'ClusteringParser', f'{key}:metrics')
    ParameterValidator.assert_all_type_and_value(instruction['metrics'], str, 'ClusteringParser', f'{key}:metrics')

    for metric in instruction['metrics']:
        assert hasattr(sklearn_metrics, metric), (f"Clustering parser: metric {metric} is not a valid metric. See the "
                                                  f"list of scikit-learn's metrics for clustering.")

    return instruction['metrics']


def parse_clustering_settings(key: str, instruction: dict, symbol_table: SymbolTable) -> List[ClusteringSetting]:
    ParameterValidator.assert_type_and_value(instruction['clustering_settings'], list, 'ClusteringParser',
                                             f'{key}:clustering_settings')
    valid_encodings = symbol_table.get_keys_by_type(SymbolType.ENCODING)
    valid_dim_red = [method.symbol for method in symbol_table.get_by_type(SymbolType.ML_METHOD)
                     if isinstance(method.item, DimRedMethod)]
    valid_clusterings = [method.symbol for method in symbol_table.get_by_type(SymbolType.ML_METHOD)
                         if isinstance(method.item, ClusteringMethod)]

    settings_objs = []
    for setting in instruction['clustering_settings']:
        setting_obj = make_setting_obj(setting, valid_encodings, valid_clusterings, valid_dim_red, symbol_table,
                                       instruction)
        settings_objs.append(setting_obj)

    unique_objs = list(set(settings_objs))

    if len(unique_objs) < len(settings_objs):
        logging.warning(f"Clustering parser: clustering_settings contains {len(settings_objs) - len(unique_objs)} "
                        f"duplicate settings, keep the following: {[obj.get_key() for obj in unique_objs]}")

    return unique_objs


def make_setting_obj(setting, valid_encodings, valid_clusterings, valid_dim_red, symbol_table, instruction):
    ParameterValidator.assert_keys_present(setting.keys(), ['encoding', 'method'], 'ClusteringParser',
                                           'clustering_settings')
    ParameterValidator.assert_keys(setting.keys(), ['encoding', 'dim_reduction', 'method'], 'ClusteringParser',
                                   'clustering_settings', exclusive=False)

    ParameterValidator.assert_in_valid_list(setting['encoding'], valid_encodings, 'ClusteringParser', 'encoding')
    ParameterValidator.assert_in_valid_list(setting['method'], valid_clusterings, 'ClusteringParser',
                                            'method')
    if 'dim_reduction' in setting and setting['dim_reduction'] is not None:
        ParameterValidator.assert_in_valid_list(setting['dim_reduction'], valid_dim_red, 'ClusteringParser',
                                                'dim_reduction')
        dim_reduction = copy.deepcopy(symbol_table.get(setting['dim_reduction']))
        dim_red_params = symbol_table.get_config(setting['dim_reduction'])
        dim_red_name = setting['dim_reduction']
    else:
        dim_reduction, dim_red_params, dim_red_name = None, None, None

    encoder = make_encoder_obj(symbol_table, setting['encoding'], instruction['dataset'])
    method = copy.deepcopy(symbol_table.get(setting['method']))

    return ClusteringSetting(encoder=encoder,
                             encoder_params=symbol_table.get_config(setting['encoding'])['encoder_params'],
                             encoder_name=setting['encoding'], clustering_method=method,
                             clustering_params=symbol_table.get_config(setting['method']),
                             clustering_method_name=setting['method'], dim_reduction_method=dim_reduction,
                             dim_red_params=dim_red_params, dim_red_name=dim_red_name)


def make_encoder_obj(symbol_table, encoding_key, dataset_key):
    return symbol_table.get(encoding_key).build_object(symbol_table.get(dataset_key),
                                                       **symbol_table.get_config(encoding_key)["encoder_params"]) \
        .set_context({"dataset": symbol_table.get(dataset_key)})
