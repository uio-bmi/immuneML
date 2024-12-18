Troubleshooting
===============

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML troubleshooting
   :twitter:description: See some commonly asked questions related to immuneML installation or notification at runtime.
   :twitter:image: https://docs.immuneml.uio.no/_images/receptor_classification_overview.png


.. toctree::
   :maxdepth: 2

Installation issues
-------------------



immuneML no longer supports Python version 3.8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
From immuneML version 3, Python 3.8 and lower are no longer supported. Please use Python version 3.9 or higher.
immuneML has been tested extensively with Python version 3.11.

During installation of the dependency pystache, I get an error: use_2to3 is invalid.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The full error reads:

.. code-block:: console

    pystache: using: version '58.0.4' of <module 'setuptools' from 'immuneml_env/lib/python3.8/site-packages/setuptools/__init__.py'>
    Warning: 'classifiers' should be a list, got type 'tuple'
    error in pystache setup command: use_2to3 is invalid.

This issue occurs due to an incompatibility between pystache and newer version of setuptools (known to occur with setuptools version 58.0.4 and higher).
A temporary workaround is to use an older version of setuptools, for example version 50.3.2.
Updates on fixing the incompatibility in pystache can be followed in this GitHub issue: https://github.com/pypa/pypi-support/issues/1422

When installing all requirements from requirements.txt, there is afterward an error with yaml package (No module named yaml)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This issue might be helpful: https://github.com/yaml/pyyaml/issues/291. Try installing yaml manually with a specific version.

Runtime issues
--------------

When running immuneML, I get the error "AttributeError: 'DataFrame' object has no attribute 'map'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Pandas dataframe 'map' is a feature that was introduced in pandas 2.1. Please make sure your pandas version is at least 2.1.
Note that this pandas version requires at least Python version 3.9.

When running immuneML, I get the error "cannot import name 'triu' from 'scipy.linalg'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This issue occurs due to the function 'triu' being removed from scipy in version 1.13, which
is called by the dependency gensim. The workaround is to use a lower scipy version, such as 1.12.
In the next release of gensim this issue will be fixed. See also: https://github.com/piskvorky/gensim/issues/3525


When running immuneML, I get "ModuleNotFoundError: No module named 'init'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This issue occurs due to an incompatibility between pystache and newer version of setuptools (known to occur with setuptools version 58.0.4 and higher).
A temporary workaround is to use an older version of setuptools, for example version 50.3.2.
Updates on fixing the incompatibility in pystache can be followed in this GitHub issue: https://github.com/pypa/pypi-support/issues/1422

When running the TrainMLModel instruction multiple times, sometimes it fails saying that there is only one class in the data. Why does this happen?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please check the number of examples used for machine learning (e.g. number of repertoires or receptors). If there are very few examples, and/or if classes
are not balanced, it is possible that just by chance, the data from only one class will be in the training set. If that happens, the classifiers
will not train and an error will be thrown. To fix this, try working with a larger dataset or check how TrainMLModel is specified.
If TrainMLModel does nested cross-validation, it might require a bit more data. To perform only cross-validation, under `selection` key, specify
that `split_strategy` is `random` and that `training_percentage` is `1` (to use all data from the inner loop for training). In this way, instead of having
multiple training/validation/test splits, there will be only training/test splits as specified under key `assessment` in TrainMLModel instruction.

When running DeepRC I get TypeError: can't concat str to bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This error occurs when h5py version 3 or higher is used. Try using version 2.10.0 or lower.

I am trying to run a report, but it gives no results and no errors. What happened?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To ensure that large analyses do not crash if one of the reports failed (for example, if an error occurs
during calculating the results or plotting), reports are run in a 'safe mode' so that errors do not stop the execution.
Check the output log.txt file to see if any errors or warnings were produced by the reports you tried to run.
