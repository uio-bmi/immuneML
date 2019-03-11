from source.workflows.steps.Step import Step


class ReportGenerator(Step):
    @staticmethod
    def run(input_params: dict = None):
        ReportGenerator.perform_step(input_params)

    @staticmethod
    def check_prerequisites(input_params: dict = None):
        pass

    @staticmethod
    def perform_step(input_params: dict = None):
        for key in input_params["reports"].keys():

            report_obj = input_params["reports"][key]["report"]
            params = {}

            for k in ["dataset", "result_path", "batch_size"]:
                if k in input_params:
                    params[k] = input_params[k]

            # TODO: make this more general and add other parameters which might be needed for other report types
            # if some parameters are set by user, it will overwrite what was set by default program behaviour
            # when the dicts are merged in order params, report_config["params"]
            report_obj.generate_report({**params, **input_params["reports"][key]["params"]})
