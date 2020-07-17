from enum import Enum


class PanelLayoutType(Enum):

    GRID = "panels are arranged by grouping rows according to `row_grouping_labels` and grouping columns according to " \
           "`column_grouping_labels`"
    WRAP = "panels are arranged according to the number of rows/columns specified, rather than based on a grouping " \
           "variable - note that in this case, `row_grouping_labels` and `column_grouping_labels` are treated " \
           "equivalently and `panel_nrow` and `panel_ncol` should be set to the desired values, or left empty for " \
           "default layout"
