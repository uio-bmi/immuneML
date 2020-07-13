from enum import Enum


class PanelLabelSwitchType(Enum):

    NULL = "labels for panels in each row are displayed on the right side, and labels for panels in each column are " \
           "displayed at the top"
    X = "labels for panels in each row are displayed on the left side, and labels for panels in each column are " \
        "displayed at the top"
    Y = "labels for panels in each row are displayed on the right side, and labels for panels in each column are " \
        "displayed at the bottom"
    BOTH = "labels for panels in each row are displayed on the left side, and labels for panels in each column are " \
           "displayed at the bottom"
