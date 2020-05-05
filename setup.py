import glob

from setuptools import setup, find_packages

setup(
    name="immune-ml",
    version="0.0.20",
    description="immuneML is a software platform for machine learning analysis of immune receptor sequences",
    long_description=open("README.md").read(),
    author="Milena Pavlovic",
    author_email="milenpa@student.matnat.uio.no",
    url="https://github.com/uio-bmi/ImmuneML",
    install_requires=['rpy2', "pytest==4.3.1", 'pandas==0.24.2', 'scikit-learn==0.20.3', 'gensim==3.8.1',
                      'matplotlib==3.1.1', 'editdistance==0.5.3', 'dask[complete]',
                      'regex', 'tzlocal', 'airr==1.2.1', 'pystache==0.5.4'],
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.6',
    packages=find_packages(exclude=["test", "test.*", "performance_tests", "performance_tests.*"]),
    package_data={
        'source': ['IO/dataset_import/conversion/*.csv', "presentation/html/templates/*.html", "presentation/html/templates/css/*.css"] +
                  [f"config/default_params/{dir_name.split('/')[-1]}/*.yaml" for dir_name in glob.glob("./source/config/default_params/*")]
    },
    entry_points={
        'console_scripts': [
            'immune-ml = source.app.ImmuneMLApp:main'
        ]
    },
)
