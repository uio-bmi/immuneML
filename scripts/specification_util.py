from pathlib import Path
from scripts.DocumentatonFormat import DocumentationFormat


def update_docs_per_mapping(docs: str, mapping: dict):
    for key in mapping:
        docs = docs.replace(key, mapping[key])
    return docs


def write_class_docs(doc_format: DocumentationFormat, file):
    title = f"\n{doc_format.cls_name}\n{doc_format.level_heading}\n\n" if doc_format.cls_name != "" else "\n\n"
    file.writelines(title)
    if hasattr(doc_format.cls, "get_documentation"):
        file.writelines(doc_format.cls.get_documentation())
    elif doc_format.cls.__doc__ is not None:
        file.writelines(doc_format.cls.__doc__)


def make_docs(path: Path, classes, filename, drop_name_part, file_open_mode="w"):
    classes.sort(key=lambda cls: cls.__name__)
    classes_to_document = [DocumentationFormat(cls, cls.__name__.replace(drop_name_part, ""), DocumentationFormat.LEVELS[1])
                           for cls in classes]

    file_path = path / filename
    with file_path.open(file_open_mode) as file:
        for doc_format in classes_to_document:
            write_class_docs(doc_format, file)
