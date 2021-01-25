from pathlib import Path

import pystache


class TemplateParser:

    @staticmethod
    def parse(template_path: Path, template_map: dict, result_path: Path) -> Path:
        with template_path.open("r") as template_file:
            template = template_file.read()

        rendered_template = pystache.render(template, template_map)

        with result_path.open("w") as file:
            file.writelines(rendered_template)

        return result_path
