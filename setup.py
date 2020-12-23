import glob

from setuptools import setup, find_packages

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
