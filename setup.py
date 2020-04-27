from setuptools import setup, find_packages

setup(
    name="immune-ml",
    version="0.0.4",
    description="immuneML is a software platform for machine learning analysis of immune receptor sequences",
    long_description=open("README.md").read(),
    author="Milena Pavlovic",
    author_email="milenpa@student.matnat.uio.no",
    url="https://github.com/uio-bmi/ImmuneML",
    install_requires=['rpy2', "pytest==4.3.1", 'pandas==0.24.2', 'scikit-learn==0.20.3', 'gensim==3.8.1',
                      'matplotlib==3.1.1', 'editdistance==0.5.3', 'dask[complete]',
                      'regex', 'tzlocal', 'airr==1.2.1'],
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.6',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'immune-ml = source.app.ImmuneMLApp:main'
        ]
    },
)
