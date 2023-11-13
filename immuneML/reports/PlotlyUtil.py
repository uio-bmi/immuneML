import plotly.graph_objects as go


class PlotlyUtil:

    @staticmethod
    def add_single_axis_labels(figure, x_label, y_label, x_label_position, y_label_position):
        """
        Takes a multi-facet plotly figure and replaces the repetitive x and y axis labels
        with single axis labels in the form of annotations.

        Args:
            figure: a plotly figure
            x_label: the x label text
            y_label: the y label text
            x_label_position: the position of the new axis labels relative to the respective axes
            y_label_position: the position of the new axis labels relative to the respective axes

        Returns:
             an updated plotly figure

        """
        # hide subplot y-axis titles and x-axis titles
        for axis in figure.layout:
            if type(figure.layout[axis]) == go.layout.YAxis:
                figure.layout[axis].title.text = ''
            if type(figure.layout[axis]) == go.layout.XAxis:
                figure.layout[axis].title.text = ''

        # keep all other annotations and add single y-axis and x-axis title:
        figure.update_layout(
            # keep the original annotations and add a list of new annotations:
            annotations=list(figure.layout.annotations) +
                        [go.layout.Annotation(x=y_label_position, y=0.5, font={"size": 14},
                                              showarrow=False, text=y_label, textangle=-90, xref="paper",
                                              yref="paper")] +
                        [go.layout.Annotation(x=0.5, y=x_label_position, font={"size": 15},
                                              showarrow=False, text=x_label, textangle=-0,
                                              xref="paper", yref="paper")])
        return figure

    @staticmethod
    def get_amino_acid_color_map():
        '''To be used whenever plotting for example a barplot where each amino acid is represented,
        to be used as a value for plotly's color_discrete_map'''
        return {'Y': 'rgb(102, 197, 204)', 'W': 'rgb(179,222,105)', 'V': 'rgb(220, 176, 242)',
                'T': 'rgb(217,217,217)', 'S': 'rgb(141,211,199)', 'R': 'rgb(251,128,114)',
                'Q': 'rgb(158, 185, 243)', 'P': 'rgb(248, 156, 116)', 'N': 'rgb(135, 197, 95)',
                'M': 'rgb(254, 136, 177)', 'L': 'rgb(201, 219, 116)', 'K': 'rgb(255,237,111)',
                'I': 'rgb(180, 151, 231)', 'H': 'rgb(246, 207, 113)', 'G': 'rgb(190,186,218)',
                'F': 'rgb(128,177,211)', 'E': 'rgb(253,180,98)',  'D': 'rgb(252,205,229)',
                'C': 'rgb(188,128,189)', 'A': 'rgb(204,235,197)'}