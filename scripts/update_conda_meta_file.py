from pathlib import Path


def update_dependencies(file_content):
    new_file_content = []

    for line in file_content:
        if line.startswith("  license_file:"):
            new_file_content.append("  license_file: https://github.com/uio-bmi/immuneML/blob/master/LICENSE.md\n")
        elif line.startswith("    - torch"):
            new_file_content.append(line.replace("torch", "pytorch"))
        elif line.startswith("  doc_url:"):
            new_file_content.append("  doc_url: https://docs.immuneml.uio.no\n")
        elif line.startswith("          environment-file:"):
            new_file_content.append("          environment-file: immuneML-dev/meta.yaml\n")
        else:
            new_file_content.append(line)

    return new_file_content


def main():
    meta_file_path = Path("immuneml-dev/meta.yaml")
    with meta_file_path.open('r') as file:
        file_content = file.readlines()

    file_content = update_dependencies(file_content)

    with meta_file_path.open('w') as file:
        file.writelines(file_content)


if __name__ == "__main__":
    main()
