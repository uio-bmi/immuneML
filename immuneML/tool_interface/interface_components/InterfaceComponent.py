from abc import ABC
import subprocess
import json
import time
import sys


class InterfaceComponent(ABC):

    @staticmethod
    def produce_JSON_object(**input_data):
        """Produces a JSON object from input data
        Returns JSON object on success, or None if error
        """

        try:
            json_bytes = json.dumps(input_data)
        except Exception as e:
            print(f"Error: {e}")
            return None

        return json_bytes

    @staticmethod
    def execution_animation(process: subprocess):
        """Function creates an animation to give user feedback while process in running
        """
        num_dots = 0

        while process.poll() is None:
            sys.stdout.write('\rRunning subprocess{}   '.format('.' * num_dots))
            sys.stdout.flush()
            num_dots = (num_dots + 1) % 4
            time.sleep(0.5)

        sys.stdout.flush()
        sys.stdout.write("\rSubprocess finished")

    @staticmethod
    def show_process_output(ml_specs: dict):
        """ Returns true or false for showing process output based on YAML spec file
        """
        show_output = False
        value = ml_specs.get("show_process_output")

        if value is not None:
            if value.lower() == "true":
                show_output = True
            elif value.lower() != "false":
                print(f"Show_process_output must have parameter 'true' or 'false'. Parameter given: {value}")
            else:
                print("""Process output not showing. Include 'Show_process_output: true' in the YAML spec file to 
                see process output""")

        return show_output
