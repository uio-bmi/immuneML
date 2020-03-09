from enum import Enum


class ErrorBarMeaning(Enum):
    STANDARD_DEVIATION = "standard_deviation"
    STANDARD_ERROR = "standard_error"
    CONFIDENCE_INTERVAL = "confidence_interval"
