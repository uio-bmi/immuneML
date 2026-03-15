



ApplyGenModel instruction implements applying generative AIRR models on the sequence level.

This instruction takes as input a trained model (trained in the :ref:`TrainGenModel` instruction)
which will be used for generating data and the number of sequences to be generated.
It can also produce reports of the applied model and reports of generated sequences.


**Specification arguments:**

- gen_examples_count (int): how many examples (sequences, repertoires) to generate from the applied model

- reports (list): list of report ids (defined under definitions/reports) to apply after generating
  gen_examples_count examples; these can be data reports (to be run on generated examples), ML reports (to be run
  on the fitted model)

- ml_config_path (str): path to the trained model in zip format (as provided by TrainGenModel instruction)

**YAML specification:**

.. highlight:: yaml
.. code-block:: yaml

    instructions:
        my_apply_gen_model_inst: # user-defined instruction name
            type: ApplyGenModel
            gen_examples_count: 100
            ml_config_path: ./config.zip
            reports: [data_rep1, ml_rep2]


