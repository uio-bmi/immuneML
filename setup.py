from pathlib import Path

from setuptools import setup, find_packages

from immuneML.environment.Constants import Constants


def import_requirements(filename) -> list:
    with open(filename, 'r') as file:
        requirements = file.read().split("\n")
    return requirements


setup(
    name="immuneML_dev",
    version=Constants.DEV_VERSION,
    description="immuneML is a software platform for machine learning analysis of immune receptor repertoires.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="immuneML Team",
    author_email="milenpa@student.matnat.uio.no",
    url="https://github.com/uio-bmi/immuneML",
    install_requires=["numpy<=1.23.5", "pytest>=4", "pandas>=1,<2.0", "PyYAML>=5.3", "scikit-learn>=0.23",
                      "gensim>=4", "matplotlib>=3.1", "editdistance", "regex", "tzlocal", "airr>=1,<1.4",
                      "fishersapi", "pystache", "torch>=1.5.1", "dill>=0.3", "plotly>=4", "logomaker>=0.8",
                      "matplotlib-venn>=0.11", "scipy", "bionumpy"],
    extras_require={
        "TCRdist": ["parasail==1.2", "tcrdist3>=0.1.6"],
        "gen_models": ['olga', 'sonnia']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3"
    ],
    python_requires='>=3.7',
    packages=find_packages(exclude=["test", "test.*"]),
    package_data={
        'immuneML': [str(Path('IO/dataset_import/conversion/*.csv')),
                     str(Path("presentation/html/templates/*.html")),
                     str(Path("presentation/html/templates/css/*.css")),
                     str(Path('encodings/atchley_kmer_encoding/*.csv'))] +
                    [str(Path("config/default_params/") / dir.name / "*.yaml") for dir in Path("./immuneML/config/default_params/").glob("*")],
    },
    entry_points={
        'console_scripts': [
            'immune-ml = immuneML.app.ImmuneMLApp:main',
            'immune-ml-quickstart = immuneML.workflows.instructions.quickstart:main'
        ]
    },
)
