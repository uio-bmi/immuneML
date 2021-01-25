import plotly.graph_objects as go

class PlotlyUtil:

    @staticmethod
    def add_single_axis_labels(figure, x_label, y_label, x_label_position, y_label_position):
        '''
        Takes a multi-facet plotly figure and replaces the repetitive x and y axis labels
        with single axis labels in the form of annotations.

        :param figure: a plotly figure
        :param label_position: the position of the new axis labels relative to the respective axes
        :param x_label: the x label text
        :param y_label: the y label text
        :return: an updated plotly figure
        '''
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