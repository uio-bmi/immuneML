from pathlib import Path

from setuptools import setup, find_packages

from source.environment.Constants import Constants


def import_requirements(filename) -> list:
    with open(filename, 'r') as file:
        requirements = file.read().split("\n")
    return requirements


setup(
    name="immuneML",
    version=Constants.VERSION,
    description="immuneML is a software platform for machine learning analysis of immune receptor repertoires.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Milena Pavlovic",
    author_email="milenpa@student.matnat.uio.no",
    url="https://github.com/uio-bmi/immuneML",
    install_requires=import_requirements("requirements.txt"),
    extras_require={
        "TCRDist": import_requirements("requirements_TCRdist.txt")
    },
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.7',
    packages=find_packages(exclude=["test", "test.*", "performance_tests", "performance_tests.*"]),
    package_data={
        'source': [str(Path('IO/dataset_import/conversion/*.csv')),
                   str(Path("presentation/html/templates/*.html")),
                   str(Path("presentation/html/templates/css/*.css")),
                   str(Path('encodings/atchley_kmer_encoding/*.csv'))] +
                  [str(Path("config/default_params/") / dir.name / "*.yaml") for dir in Path("./source/config/default_params/").glob("*")],
    },
    entry_points={
        'console_scripts': [
            'immune-ml = source.app.ImmuneMLApp:main',
            'immune-ml-quickstart = source.workflows.instructions.quickstart:main'
        ]
    },
)
