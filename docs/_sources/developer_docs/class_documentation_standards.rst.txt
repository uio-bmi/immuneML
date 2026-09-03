Class documentation should be added as a docstring to all new Encoder, MLMethod, Report or Preprocessing classes.
The class docstrings are used to automatically generate the documentation web pages, using Sphinx `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_, and should adhere to a standard format:


#. A short, general description of the functionality

#. Optional extended description, including any references or specific cases that should be considered. For instance: if a class can only be used for a particular dataset type. Compatibility between Encoders, MLMethods and Reports should also be described.

#. For encoders, the appropriate dataset type(s). For example:

.. code-block:: yaml

     **Dataset type:**

     - SequenceDatasets

     - RepertoireDatasets

#. A list of arguments, when applicable. This should follow the format below:

.. code-block:: yaml

      **Specification arguments:**

      - parameter_name (type): a short description

      - other_paramer_name (type): a short description

#. A YAML snippet, to show an example of how the new component should be called. Make sure to test your YAML snippet in an immuneML run to ensure it is specified correctly. The following formatting should be used to ensure the YAML snippet is rendered correctly:

.. code-block:: yaml

      **YAML specification:**

      .. code-block:: yaml

          definitions:
              yaml_keyword: # could be encodings/ml_methods/reports/etc...
                  my_new_class:
                      MyNewClass:
                          parameter_name: 0
                          other_paramer_name: 1
