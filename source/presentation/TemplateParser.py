import pystache


class TemplateParser:

    @staticmethod
    def parse(template_path: str, template_map: dict, result_path: str) -> str:
        with open(template_path, "r") as template_file:
            template = template_file.read()

        rendered_template = pystache.render(template, template_map)

        with open(result_path, "w") as file:
            file.writelines(rendered_template)

        return result_path
