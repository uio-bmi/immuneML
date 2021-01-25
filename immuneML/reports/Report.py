import abc
import logging

from immuneML.reports.ReportResult import ReportResult


class Report(metaclass=abc.ABCMeta):
    """
    This class defines what report classes should look like: they all have to inherit this class and implement the abstract methods: build_object()
    from parameters and generate() the report once all properties are set (in immuneML this will be taken care of by the instructions). If there are
    any prerequisites needed to run the report (e.g., check if all parameter values are properly set), the check_prerequisites function should be
    overwritten to reflect that and determine if everything is set before generate() is run. See specific functions for more details.

    """

    def __init__(self, name: str = None):
        self.name = name

    @classmethod
    @abc.abstractmethod
    def build_object(cls, **kwargs):
        """
        Creates the object of the subclass of the Report class from the parameters so that it can be used in the analysis. Depending on the type of
        the report, the parameters provided here will be provided in parsing time, while the other necessary parameters (e.g., subset of the data from
        which the report should be created) will be provided at runtime. For more details, see specific direct subclasses of this class, describing
        different types of reports.

        Args:

            **kwargs: keyword arguments that will be provided by users in the specification (if immuneML is used as a command line tool) or in the
             dictionary when calling the method from the code, and which should be used to create the report object

        Returns:

            the object of the appropriate report class

        """
        pass

    @abc.abstractmethod
    def _generate(self) -> ReportResult:
        """
        The function that needs to be implemented by the Report subclasses which actually creates the report (figures, tables, text files), depending
        on the specific aim of the report. After checking all prerequisites (e.g., if all parameters were set properly), generate_report() will call
        this function and return its result.

        Returns:

            ReportResult object which encapsulates all outputs (figure, table, and text files) so that they can be conveniently linked to in the
            final output of instructions

        """
        pass

    def check_prerequisites(self) -> bool:
        """
        Checks prerequisites for the generation of the report of specific class (e.g., if the class of the MLMethod instance is the one required by
        the report, if the data has been encoded to make a report of encoded dataset). In the instructions in immuneML, this function is used to
        determine whether to call generate_report() in the specific situation. Each report subclass has its own set of prerequisites. If the report
        cannot be run, the information on this will be logged and the report skipped in the specific situation. No error will be raised. See
        subclasses of the class :py:obj:`~immuneML.workflows.instructions.Instruction.Instruction` for more information on how the reports are executed.

        Returns:
             boolean value True if the prerequisites are o.k., and False otherwise.
        """
        return True

    def set_context(self, context: dict):
        """
        Context is a dictionary with information that is accessible from the level of instruction and can be used to precompute certain values that
        can be later reused to speed up the generation of the subsequent reports of the same time. For instance, if one should compute the distance
        between all repertoires based on the sequence content, it is possible to store the full dataset in the context, compute the distances on the
        full dataset and then only extract the distances need for the current dataset in the later calls (e.g., when training dataset is passed as
        input). Only some reports will need this functionality.

        Warning: It is very important to be careful when using the context to avoid leaking the information between training and test datasets.

        Args:

            context (dict): a dictionary where the values are variables that are typically only available on the top-level of the instruction, and
                which are used to precompute results in order to speed up subsequent generation of the same report on subsets of those values.

        Returns:

            self - so that it can be chained with the other function calls

        """
        return self

    def generate_report(self) -> ReportResult:
        """
        Generates a report of the given class if the prerequisites are satisfied. It handles all exceptions so that if there is an error while
        generating a report, the execution of the rest of the code (e.g., more time-expensive parts, like instructions) is not influenced.

        Returns:

            ReportResult object which encapsulates all outputs (figure, table, and text files) so that they can be conveniently linked to in the
            final output of instructions

        """
        try:
            if self.check_prerequisites():
                return self._generate()
        except Exception as e:
            logging.exception(f"An exception occurred while generating report {self.name}. See the details below:")
            logging.warning(f"Report {self.name} encountered an error and could not be generated: {e}.")

    def _safe_plot(self, output_written=True, **kwargs):
        """
        A wrapper around the function _plot() which catches any error that may be thrown by this function (e.g. errors in R),
        and shows an informative warning message instead. This is to prevent immuneML from crashing when the analysis has been
        completed but only a figure could not be plotted.

        Args:

            output_written: indicates whether the output was written to a file, this changes the error message.

            kwargs: passed to _plot()

        Returns:

            the results of _plot(), typically a ReportOutput object

        """
        warning_mssg = f"{self.__class__.__name__}: an error occurred when attempting to plot the data. \n"
        if output_written:
            warning_mssg += "\nThe data has been written to a file, but no plot has been created."
        else:
            warning_mssg += "\nNo plot has been created."
        try:
            if callable(getattr(self, '_plot', None)):
                return self._plot(**kwargs)
        except Exception as e:
            logging.exception(f"An exception occurred while plotting the data in report {self.name}. See the details below:")
            logging.warning(warning_mssg)
