# -*- coding: utf-8 -*-
# Copyright (c) 2020, immuneML Development Team.
# Distributed under the LGPLv3 License. See LICENSE for more info.
"""immuneML.

This file is part of immneML.

immuneML is free software, please our term and conditions detailed
in the file LIVENCE.md
"""
import ast
import glob
import pathlib
from codecs import open as openc
from setuptools import setup, find_packages

FULL_VERSION = '0.9.0.dev0'  # Automatically set by setup_version.py


def get_long_description():
    """Return the contents of README.md as a string."""
    here = pathlib.Path(__file__).absolute().parent
    long_description = ''
    with openc(here.joinpath('README.md'), encoding='utf-8') as fileh:
        long_description = fileh.read()
    return long_description


def get_version():
    """Return the version from version.py as a string."""
    here = pathlib.Path(__file__).absolute().parent
    filename = here.joinpath('source', 'version.py')
    with openc(filename, encoding='utf-8') as fileh:
        for lines in fileh:
            if lines.startswith('FULL_VERSION ='):
                version = ast.literal_eval(lines.split('=')[1].strip())
                return version
    return FULL_VERSION


def import_requirements(filename) -> list:
    """Import requirements."""
    with open(filename, 'r') as file:
        requirements = file.read().split("\n")
    return requirements


setup(
    name="immuneML",
    version=get_version(),
    description="immuneML is a software platform for machine learning analysis of immune receptor sequences",
    long_description=get_long_description(),
    author="Milena Pavlovic",
    author_email="milenpa@student.matnat.uio.no",
    url="https://github.com/uio-bmi/ImmuneML",
    install_requires=import_requirements("requirements.txt"),
    extras_require={
        "R_plots":  import_requirements("requirements_R_plots.txt"),
        "TCRDist": import_requirements("requirements_TCRdist.txt"),
        # "DeepRC": import_requirements("requirements_DeepRC.txt"),  # No longer supported since pip 19
    },
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.7',
    packages=find_packages(exclude=["test", "test.*", "performance_tests", "performance_tests.*"]),
    package_data={
        'source': ['IO/dataset_import/conversion/*.csv', "presentation/html/templates/*.html", "presentation/html/templates/css/*.css",
                   "visualization/*.R", "visualization/*.r", 'encodings/atchley_kmer_encoding/*.csv'] +
                  [f"config/default_params/{dir_name.split('/')[-1]}/*.yaml" for dir_name in
                   glob.glob("./source/config/default_params/*")],
        'datasets': [path.rsplit("datasets/")[1] for path in glob.glob("datasets/**/*.tsv", recursive=True)] +
                    [path.rsplit("datasets/")[1] for path in glob.glob("datasets/**/*.csv", recursive=True)]
    },
    entry_points={
        'console_scripts': [
            'immune-ml = source.app.ImmuneMLApp:main'
        ]
    },
)
