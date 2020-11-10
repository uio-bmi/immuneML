import glob

from setuptools import setup, find_packages

setup(
    name="immune-ml",
    version="0.0.91",
    description="immuneML is a software platform for machine learning analysis of immune receptor sequences",
    long_description=open("README.md").read(),
    author="Milena Pavlovic",
    author_email="milenpa@student.matnat.uio.no",
    url="https://github.com/uio-bmi/ImmuneML",
    install_requires=["pytest>=4.3.1", "pandas>=1.1.1", "scikit-learn>=0.22.2.post1", "gensim==3.8.1", "matplotlib>=3.1.1", "editdistance==0.5.3",
                      "dask[complete]", "regex", "tzlocal", "airr==1.2.1", "pystache==0.5.4", "torch==1.5.1", "numpy>=1.18.2", "h5py>=2.9.0",
                      "dill>=0.3.0", "tqdm>=0.24.2", "logomaker>=0.8", "plotly>=4.8.2", "fishersapi", "requests>=2.21.0",
                      "deeprc@git+https://github.com/ml-jku/DeepRC@fec4b4f4b2cd70e00e8de83da169560dec73a419", "matplotlib-venn>=0.11.6",
                      "widis-lstm-tools@git+https://github.com/widmi/widis-lstm-tools", "tcrdist3>=0.1.6"],
    extras_require={
        "R_plots":  ["rpy2"]
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
