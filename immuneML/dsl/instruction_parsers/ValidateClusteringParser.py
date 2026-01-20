import inspect
import logging
import shutil
import tempfile
from pathlib import Path
from typing import List

import sklearn.metrics as sklearn_metrics

from immuneML.IO.ml_method.ClusteringImporter import ClusteringImporter
from immuneML.reports.clustering_method_reports.ClusteringMethodReport import ClusteringMethodReport
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.data_model.SequenceParams import RegionType
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.dsl.instruction_parsers.LabelHelper import LabelHelper
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.clustering.ValidateClusteringInstruction import ValidateClusteringInstruction


class ValidateClusteringParser:

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> ValidateClusteringInstruction:
        valid_keys = ['type', 'clustering_config_path', 'dataset', 'metrics', 'validation_type', 'labels',
                      'sequence_type', 'region_type', 'number_of_processes', 'reports']

        ParameterValidator.assert_keys(instruction.keys(), valid_keys, ValidateClusteringParser.__name__, key)
        ParameterValidator.assert_keys_present(instruction.keys(), ['clustering_config_path', 'dataset', 'metrics', 'validation_type'],
                                                ValidateClusteringParser.__name__, key)

        ParameterValidator.assert_region_type(instruction, ValidateClusteringParser.__name__)
        ParameterValidator.assert_sequence_type(instruction, ValidateClusteringParser.__name__)

        # Load the clustering item from the exported zip file
        clustering_item = self._load_clustering_item(instruction['clustering_config_path'])

        # Get the validation dataset
        dataset = symbol_table.get(instruction['dataset'])

        # Parse metrics
        metrics = self._parse_metrics(key, instruction)

        # Parse validation types
        validation_type = self._parse_validation_type(key, instruction)

        # Parse labels if provided
        label_config = self._parse_labels(key, instruction, dataset)

        # Get optional parameters with defaults
        number_of_processes = instruction.get('number_of_processes', 1)

        # Parse reports
        reports = self._parse_reports(instruction, symbol_table)

        return ValidateClusteringInstruction(
            clustering_item=clustering_item,
            dataset=dataset,
            metrics=metrics,
            validation_type=validation_type,
            label_config=label_config,
            sequence_type=SequenceType[instruction['sequence_type'].upper()],
            region_type=RegionType[instruction['region_type'].upper()],
            number_of_processes=number_of_processes,
            reports=reports
        )

    def _load_clustering_item(self, config_path: str):
        """Load a ClusteringItem from an exported zip file or directory."""
        config_path = Path(config_path)

        if config_path.suffix == '.zip':
            # Extract zip to temp directory and load
            temp_dir = tempfile.mkdtemp()
            shutil.unpack_archive(config_path, temp_dir)
            cl_item, config = ClusteringImporter.import_clustering_item(Path(temp_dir))
            # Note: temp_dir is not cleaned up here to keep the loaded objects valid
            # It will be cleaned up when the process ends
        else:
            # Assume it's a directory
            cl_item, config = ClusteringImporter.import_clustering_item(config_path)

        return cl_item

    def _parse_metrics(self, key: str, instruction: dict) -> List[str]:
        """Parse and validate clustering metrics."""
        ParameterValidator.assert_type_and_value(instruction['metrics'], list, 'ValidateClusteringParser', f'{key}:metrics')
        ParameterValidator.assert_all_type_and_value(instruction['metrics'], str, 'ValidateClusteringParser', f'{key}:metrics')

        for metric in instruction['metrics']:
            assert hasattr(sklearn_metrics, metric), (
                f"ValidateClusteringParser: metric {metric} is not a valid metric. "
                f"See the list of scikit-learn's metrics for clustering."
            )

        return instruction['metrics']

    def _parse_validation_type(self, key: str, instruction: dict) -> List[str]:
        """Parse and validate validation types."""
        ParameterValidator.assert_type_and_value(instruction['validation_type'], list, 'ValidateClusteringParser',
                                                  f'{key}:validation_type')

        valid_types = ['method_based', 'result_based']
        for vtype in instruction['validation_type']:
            ParameterValidator.assert_in_valid_list(vtype, valid_types, 'ValidateClusteringParser', 'validation_type')

        return instruction['validation_type']

    def _parse_labels(self, key: str, instruction: dict, dataset) -> LabelConfiguration:
        """Parse labels for external evaluation."""
        if 'labels' in instruction and instruction['labels'] is not None:
            return LabelHelper.create_label_config(instruction['labels'], dataset, key, 'labels')
        return LabelConfiguration()

    def _parse_reports(self, instruction: dict, symbol_table: SymbolTable) -> List:
        """Parse reports from the symbol table."""
        reports = []
        if 'reports' in instruction and instruction['reports'] is not None:
            for report_name in instruction['reports']:
                report = symbol_table.get(report_name)
                if any(isinstance(report, cls) for cls in [ClusteringMethodReport, DataReport, EncodingReport]):
                    reports.append(report)
                else:
                    logging.warning(f"Report {report_name} (type: {type(report)}) could not be added "
                                    f"to ValidateClusteringInstruction.")
        return reports