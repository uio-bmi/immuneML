import abc
import logging
import warnings

from source.reports.ReportResult import ReportResult


class Report(metaclass=abc.ABCMeta):
    """
    This class defines what report classes should look like: they all have to inherit this class and implement
    generate_report method
    """

    def __init__(self, name: str = None):
        self.name = name

    @classmethod
    @abc.abstractmethod
    def build_object(cls, **kwargs):
        pass

    @abc.abstractmethod
    def generate(self) -> ReportResult:
        pass

    def check_prerequisites(self):
        """
        Checks prerequisites, and sends warnings if the prerequisites are incorrect.
        Returns boolean value True if the prerequisites are o.k., and False otherwise.
        """
        return True

    def set_context(self, context: dict):
        return self

    def generate_report(self) -> ReportResult:
        try:
            if self.check_prerequisites():
                return self.generate()
        except Exception as e:
            logging.warning(f"Report {self.name} encountered an error and could not be generated.")

    def _safe_plot(self, output_written=True, **kwargs):
        """
        A wrapper around the function _plot() which catches any error that may be thrown by this function (e.g. errors in R),
        and shows an informative warning message instead. This is to prevent immuneML from crashing when the analysis has been
        completed but only a figure could not be plotted.

        :param output_written: indicates whether the output was written to a file, this changes the error message.
        :param kwargs: passed to _plot()
        :return: the results of _plot(), typically a ReportOutput object
        """
        warning_mssg = f"{self.__class__.__name__}: an error occurred when attempting to plot the data. \n"
        if output_written:
            warning_mssg += "\nThe data has been written to a file, but no plot has been created."
        else:
            warning_mssg += "\nNo plot has been created."
        try:
            return self._plot(**kwargs)
        except Exception as e:
            warnings.warn(warning_mssg)
