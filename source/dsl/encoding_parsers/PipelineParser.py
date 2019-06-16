from source.dsl.encoding_parsers.EncodingParameterParser import EncodingParameterParser
from source.dsl.encoding_parsers.EncodingParser import EncodingParser
from source.util.ReflectionHandler import ReflectionHandler


class PipelineParser(EncodingParameterParser):

    @staticmethod
    def parse(params: dict):

        initial_encoder, initial_params, initial_specs = EncodingParser.parse_encoder(params["initial_encoder"],
                                                                                      params["initial_encoder_params"])

        parsed = {
            "initial_encoder": initial_encoder,
            "initial_encoder_params": initial_params,
            "steps": PipelineParser._prepare_steps(params["steps"])
        }

        params["initial_encoder_params"] = initial_specs

        return parsed, params

    @staticmethod
    def _prepare_steps(steps: list):
        parsed_steps = []
        for step in steps:
            for key in step:
                step_class = ReflectionHandler.get_class_by_name(step[key]["type"])
                parsed_steps.append(step_class(**step[key].get("params", {})))
        assert len(steps) == len(parsed_steps), "PipelineParser: Each step accepts only one specification."
        return parsed_steps
