from setuptools import setup, find_packages
from pathlib import Path

from source.environment.Constants import Constants


def import_requirements(filename) -> list:
    with open(filename, 'r') as file:
        requirements = file.read().split("\n")
    return requirements


setup(
    name="immune-ml",
    version=Constants.VERSION,
    description="immuneML is a software platform for machine learning analysis of immune receptor sequences",
    long_description=open("README.md").read(),
    author="Milena Pavlovic",
    author_email="milenpa@student.matnat.uio.no",
    url="https://github.com/uio-bmi/ImmuneML",
    install_requires=import_requirements("requirements.txt"),
    extras_require={
        "R_plots":  import_requirements("requirements_R_plots.txt"),
        "DeepRC":  ["widis-lstm-tools@git+https://github.com/widmi/widis-lstm-tools", "deeprc@git+https://github.com/ml-jku/DeepRC@fec4b4f4b2cd70e00e8de83da169560dec73a419"],
        "TCRDist": import_requirements("requirements_TCRdist.txt"),
        "all": ["rpy2", "widis-lstm-tools@git+https://github.com/widmi/widis-lstm-tools", "deeprc@git+https://github.com/ml-jku/DeepRC@fec4b4f4b2cd70e00e8de83da169560dec73a419", "parasail==1.2", "tcrdist3>=0.1.6"],
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
                   str(Path("visualization/*.R")),
                   str(Path("visualization/*.r")),
                   str(Path('encodings/atchley_kmer_encoding/*.csv'))] +
                  [str(Path("config/default_params/") / dir.name / "*.yaml") for dir in Path("./source/config/default_params/").glob("*")],
        'datasets': [str(p.relative_to("datasets")) for pattern in ["**/*.tsv", "**/*.csv"] for p in Path("datasets").glob(pattern)]
    },
    entry_points={
        'console_scripts': [
            'immune-ml = source.app.ImmuneMLApp:main'
        ]
    },
)
