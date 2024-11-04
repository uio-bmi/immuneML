from pathlib import Path

from setuptools import setup, find_packages

from immuneML.environment.Constants import Constants


setup(
    name="immuneML",
    version=Constants.VERSION,
    description="immuneML is a software platform for machine learning analysis of immune receptor repertoires.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="immuneML Team",
    author_email="milenpa@student.matnat.uio.no",
    url="https://github.com/uio-bmi/immuneML",
    install_requires=["numpy<2.0.0", "pandas", "PyYAML>=5.3", "scikit-learn>=0.23",
                      "matplotlib>=3.1", "editdistance", "regex", "tzlocal", "airr>=1,<1.4",
                      "pystache", "dill>=0.3", "plotly>=4", "matplotlib-venn>=0.11", "scipy<1.13", "bionumpy>=0.2.31",
                      "umap-learn", "olga"],
    extras_require={
        'word2vec': ['gensim'],
        'fisher': ['fisher', 'fishersapi'],
        "TCRdist": ["tcrdist3>=0.1.6"],
        "gen_models": ['sonnia', 'torch'],
        "ligo": ['stitchr', 'IMGTgeneDL'],
        "DL": ['torch', 'keras', 'tensorflow', 'logomaker', 'gensim'],
        "KerasSequenceCNN": ["keras==2.11.0", "tensorflow==2.11.0"],
        "all": ['tcrdist3', 'sonnia', 'torch', 'stitchr', 'IMGTgeneDL', 'keras', 'tensorflow', 'fisher', 'logomaker',
                'fishersapi', 'gensim']
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
            'immune-ml-quickstart = immuneML.workflows.instructions.quickstart:main',
            'ligo = immuneML.app.LigoApp:main'
        ]
    },
)
