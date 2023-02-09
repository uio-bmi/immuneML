# -*- coding: utf8 -*-
from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='tmscoring', version='0.4.post0',
      description='Python implementation of the TMscore program',
      url='https://github.com/Dapid/tmscoring',
      author='David Menéndez Hurtado',
      author_email='davidmenhur@gmail.com',
      license='BSD 3-clause',
      packages=['tmscoring'],
      install_requires=['numpy', 'iminuit<2', 'biopython'],
      test_suite='nose.collector',
      tests_require=['nose'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=['Programming Language :: Python',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 3',
                   'Topic :: Scientific/Engineering :: Bio-Informatics',
                   'Intended Audience :: Science/Research',
                   'Development Status :: 4 - Beta',
                   'License :: OSI Approved :: BSD License']
)

