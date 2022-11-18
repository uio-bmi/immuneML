from dataclasses import dataclass

@dataclass
class DistributionParameters:
    distribution_type: str
    dataset_type: str
    label_name: str = None
    class_name: str = None