from enum import Enum


class PanelAxisScalesType(Enum):

    FREE_X = "x-axis scale is allowed to vary per panel"
    FREE_Y = "y-axis scale is allowed to vary per panel"
    FREE = "both x- and y-axis scales are allowed to vary per panel"
    FIXED = "x- and y-axis scales are the same across all panels"
