import glob

from setuptools import setup, find_packages

from source.environment.Constants import Constants


def import_requirements(filename) -> list:
    with open(filename, 'r') as file:
        requirements = file.read().split("\n")
    return requirements


setup(
    name="immuneML",
    version=Constants.VERSION,
    description="immuneML is a software platform for machine learning analysis of immune receptor sequences",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Milena Pavlovic",
    author_email="milenpa@student.matnat.uio.no",
    url="https://github.com/uio-bmi/immuneML",
    install_requires=import_requirements("requirements.txt"),
    extras_require={
        "DeepRC":  ["widis-lstm-tools@git+https://github.com/widmi/widis-lstm-tools", "deeprc@git+https://github.com/ml-jku/DeepRC@fec4b4f4b2cd70e00e8de83da169560dec73a419"],
        "TCRDist": import_requirements("requirements_TCRdist.txt"),
        "all": ["widis-lstm-tools@git+https://github.com/widmi/widis-lstm-tools", "deeprc@git+https://github.com/ml-jku/DeepRC@fec4b4f4b2cd70e00e8de83da169560dec73a419", "parasail==1.2", "tcrdist3>=0.1.6"],
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
                   glob.glob("./source/config/default_params/*")]
    },
    entry_points={
        'console_scripts': [
            'immune-ml = source.app.ImmuneMLApp:main',
            'immune-ml-quickstart = source.workflows.instructions.quickstart:main'
        ]
    },
)
